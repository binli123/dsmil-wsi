#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageStat
import openslide
from openslide.deepzoom import DeepZoomGenerator
from skimage import io, transform, color, img_as_ubyte, exposure
from collections import OrderedDict
import dsmil as mil
import math
import cv2
import matplotlib.pyplot as plt

class WSIInference:
    def __init__(self, 
                 embedder_low_path,
                 embedder_high_path,
                 aggregator_path,
                 tile_size=224,
                 background_threshold=7,
                 base_mag=20,
                 magnifications=[0, 1],
                 detection_threshold=0.5,
                 average=False,
                 debug_model=False,
                 nonlinear=1,
                 feature_size=1024,
                 device=None):
        """Initialize WSI inference pipeline.
        
        Args:
            embedder_low_path: Path to low magnification embedder weights
            embedder_high_path: Path to high magnification embedder weights
            aggregator_path: Path to aggregator model weights
            tile_size: Size of patches to extract
            background_threshold: Threshold for background filtering
            base_mag: Base magnification of WSI
            magnifications: List of two magnification levels relative to base_mag [high_level, low_level]
            detection_threshold: Threshold for positive detection (default: 0.5)
            average: Whether to average bag and instance predictions (default: False)
            debug_model: Whether to print detailed model debugging info (default: False)
            nonlinear: Additional nonlinear operation (default: 1)
            feature_size: Size of feature vector (default: 1024)
            device: Device to run inference on ('cuda' or 'cpu', default: None - auto select)
        """
        self.tile_size = tile_size
        self.background_threshold = background_threshold
        self.base_mag = base_mag
        self.magnifications = magnifications
        self.detection_threshold = detection_threshold
        self.average = average
        self.num_classes = 1  # Hard-coded to 1 for binary classification
        self.nonlinear = nonlinear
        self.feature_size = feature_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load models
        self.embedder_low = self._load_embedder(embedder_low_path)
        self.embedder_high = self._load_embedder(embedder_high_path)
        self.aggregator = self._load_aggregator(aggregator_path, debug=debug_model)
        
    def _load_embedder(self, path):
        """Load embedder model."""
        import torchvision.models as models
        
        # Use instance normalization as in compute_feats.py
        norm = nn.InstanceNorm2d
        resnet = models.resnet18(pretrained=False, norm_layer=norm)
        num_feats = 512
        
        # Freeze all parameters
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=self.num_classes)
        
        # Load and process state dict exactly as in compute_feats.py
        state_dict_weights = torch.load(path, map_location=self.device)
        # Remove the projection head weights (last 4 items)
        for i in range(4):
            state_dict_weights.popitem()
        
        # Create new state dict with correct keys
        state_dict_init = i_classifier.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        
        i_classifier.load_state_dict(new_state_dict, strict=False)
        i_classifier = i_classifier.to(self.device)
        i_classifier.eval()
        return i_classifier
        
    def _load_aggregator(self, path, debug=False):
        """Load aggregator model."""
        i_classifier = mil.FCLayer(in_size=self.feature_size, out_size=self.num_classes)
        b_classifier = mil.BClassifier(input_size=self.feature_size, output_class=self.num_classes, nonlinear=self.nonlinear)
        milnet = mil.MILNet(i_classifier, b_classifier)
        
        # Load model weights
        state_dict = torch.load(path, map_location=self.device)
        
        if debug:
            print("\nAggregator model state_dict keys:")
            for key in state_dict.keys():
                print(f"  {key}: {state_dict[key].shape}")
        
        milnet.load_state_dict(state_dict)
        milnet = milnet.to(self.device)
        milnet.eval()  # Ensure model is in evaluation mode
        
        if debug:
            # Print model parameters after loading
            print("\nAggregator model parameters after loading:")
            for name, param in milnet.named_parameters():
                print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
                
            # Print forward computation for a sample input
            print("\nTesting aggregator with random input:")
            try:
                with torch.no_grad():
                    random_input = torch.randn(10, self.feature_size).to(self.device)
                    ins_pred, bag_pred, attn, _ = milnet(random_input)
                    print(f"  Instance prediction shape: {ins_pred.shape}")
                    print(f"  Bag prediction shape: {bag_pred.shape}")
                    print(f"  Bag prediction value: {bag_pred.item():.4f}")
                    print(f"  Sigmoid bag prediction: {torch.sigmoid(bag_pred).item():.4f}")
            except Exception as e:
                print(f"  Error in test forward pass: {e}")
        
        return milnet
    
    def _is_background(self, tile):
        """Check if tile is background using edge detection like in deepzoom_tiler.py."""
        if tile.mode == 'RGBA':
            tile = tile.convert('RGB')
        
        # Use edge detection like in deepzoom_tiler.py
        edge = tile.filter(ImageFilter.FIND_EDGES)
        edge = ImageStat.Stat(edge).sum
        edge = np.mean(edge)/(self.tile_size**2)
        
        # # Add debug print for first few tiles
        # if not hasattr(self, '_debug_count'):
        #     self._debug_count = 0
        # if self._debug_count < 5:
        #     print(f"Tile edge score: {edge}")
        #     self._debug_count += 1
            
        return edge <= self.background_threshold
    
    def _extract_patches(self, slide_path):
        """Extract patches from WSI at both magnifications."""
        slide = openslide.OpenSlide(slide_path)
        
        # Get objective power from slide metadata
        objective_power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if objective_power is None:
            print("Warning: Could not find objective power in slide metadata, using default of 40x")
            objective_power = 40.0
        else:
            objective_power = float(objective_power)
        
        # Calculate first level based on objective power and base mag
        first_level = int(math.log2(objective_power/self.base_mag))
        
        # Create deep zoom generator
        dz = DeepZoomGenerator(slide, self.tile_size, 0)
        
        # Calculate levels exactly like deepzoom_tiler.py but account for DeepZoom's reversed level numbering
        total_levels = len(dz.level_tiles)
        level_high = total_levels - 1 - (first_level + self.magnifications[0])
        level_low = total_levels - 1 - (first_level + self.magnifications[1])
        
        # print(f"\nSlide processing information:")
        # print(f"Slide dimensions (level 0): {slide.dimensions}")
        # print(f"Objective power: {objective_power}x")
        # print(f"Base magnification: {self.base_mag}x")
        # print(f"First level at base mag: {first_level}")
        # print(f"Total available levels: {total_levels}")
        # print(f"Using high mag level: {level_high} (~{self.base_mag * (2**-self.magnifications[0])}x)")
        # print(f"Using low mag level: {level_low} (~{self.base_mag * (2**-self.magnifications[1])}x)")
        
        # Get dimensions at both levels
        high_cols, high_rows = dz.level_tiles[level_high]
        low_cols, low_rows = dz.level_tiles[level_low]
        # print(f"\nGrid dimensions:")
        # print(f"High magnification: {high_rows}x{high_cols}")
        # print(f"Low magnification: {low_rows}x{low_cols}")
        
        patches_high = []
        patches_low = []
        positions = []  # Will store high magnification positions
        
        # Dictionary to store valid low mag patches and their coordinates
        valid_low_patches = {}
        
        # First pass: identify valid low magnification patches
        # print("\nProcessing low magnification patches...")
        low_valid_count = 0
        low_total_count = 0
        
        for row in range(low_rows):
            for col in range(low_cols):
                low_total_count += 1
                try:
                    low_tile = dz.get_tile(level_low, (col, row))
                    if not self._is_background(low_tile):
                        valid_low_patches[(row, col)] = {
                            'patch': self._preprocess_tile(low_tile),
                            'name': f"{col}_{row}"
                        }
                        low_valid_count += 1
                except Exception as e:
                    print(f"Warning: Could not get low mag tile at ({col}, {row}): {e}")
                    continue
        
        # print(f"Found {low_valid_count} valid low magnification patches out of {low_total_count} total")
        
        # Second pass: get corresponding high magnification patches
        # print("\nProcessing high magnification patches...")
        scale_factor = 2  # Each low mag patch corresponds to 2x2 high mag patches
        high_valid_count = 0
        high_total_processed = 0
        
        # Track all positions for debugging
        all_positions = []
        
        for (low_row, low_col), low_info in valid_low_patches.items():
            # Calculate the corresponding high mag coordinates
            high_row_start = low_row * scale_factor
            high_col_start = low_col * scale_factor
            
            # Debug print for first few patches
            if high_valid_count < 5:
                # print(f"\nProcessing low mag patch at ({low_col}, {low_row}):")
                # print(f"Corresponding high mag region: ({high_col_start}:{high_col_start+2}, {high_row_start}:{high_row_start+2})")
                pass
            
            valid_high_patches = []
            high_positions = []
            
            # Get the 2x2 grid of high mag patches
            for i in range(scale_factor):
                for j in range(scale_factor):
                    high_row = high_row_start + i
                    high_col = high_col_start + j
                    high_total_processed += 1
                    
                    # Track all positions for debugging
                    all_positions.append((high_col, high_row))
                    
                    try:
                        high_tile = dz.get_tile(level_high, (high_col, high_row))
                        if not self._is_background(high_tile):
                            valid_high_patches.append(self._preprocess_tile(high_tile))
                            high_positions.append((high_col, high_row))
                            high_valid_count += 1
                            
                            # Debug print for first few valid high mag patches
                            if high_valid_count <= 5:
                                # print(f"Found valid high mag patch at ({high_col}, {high_row})")
                                pass
                    except Exception as e:
                        if high_valid_count < 5:
                            # print(f"Skipped high mag tile at ({high_col}, {high_row}): {e}")
                            pass
                        continue
            
            # Only use this group if we have at least one valid high mag patch
            if valid_high_patches:
                patches_high.extend(valid_high_patches)
                positions.extend(high_positions)
                patches_low.extend([low_info['patch']] * len(valid_high_patches))
        
        # Print coordinate system information
        all_cols = [p[0] for p in all_positions]
        all_rows = [p[1] for p in all_positions]
        valid_cols = [p[0] for p in positions]
        valid_rows = [p[1] for p in positions]
        
        # print(f"\nCoordinate system information:")
        # print(f"All positions - Col range: {min(all_cols)} to {max(all_cols)}")
        # print(f"All positions - Row range: {min(all_rows)} to {max(all_rows)}")
        # print(f"Valid positions - Col range: {min(valid_cols)} to {max(valid_cols)}")
        # print(f"Valid positions - Row range: {min(valid_rows)} to {max(valid_rows)}")
        
        # Check if we need to remap coordinates to match the patch naming convention
        # If your patches are named from 0 to 6, but our coordinates go higher
        if max(valid_cols) > 6 or max(valid_rows) > 6:
            # print("\nWARNING: Coordinate values exceed expected range (0-6).")
            # print("This may indicate a mismatch with your patch naming convention.")
            # print("Consider using a coordinate remapping function if needed.")
            pass
        
        # print(f"\nFinal patch statistics:")
        print(f"Low magnification patches processed: {low_total_count}")
        # print(f"Low magnification patches valid: {low_valid_count}")
        print(f"High magnification patches processed: {high_total_processed}")
        # print(f"High magnification patches valid: {high_valid_count}")
        # print(f"Position records: {len(positions)}")
        
        if len(patches_high) == 0:
            raise ValueError("No valid patches found in the slide at high magnification")
        
        assert len(patches_high) == len(patches_low), "Mismatch in number of high and low magnification patches"
        assert len(patches_high) == len(positions), "Mismatch in number of patches and positions"
            
        return patches_high, patches_low, positions
    
    def _preprocess_tile(self, tile):
        """Preprocess tile for model input."""
        if tile.mode == 'RGBA':
            tile = tile.convert('RGB')
        tile = tile.resize((self.tile_size, self.tile_size))
        tile = np.array(tile) / 255.0
        tile = torch.from_numpy(tile).permute(2, 0, 1).float().to(self.device)
        return tile
    
    def _compute_features(self, patches, embedder):
        """Compute features for patches using embedder."""
        features = []
        with torch.no_grad():
            for patch in patches:
                patch = patch.unsqueeze(0).to(self.device)
                feature, _ = embedder(patch)  # Unpack the tuple - feature and class
                features.append(feature.cpu())
        return torch.cat(features, dim=0)

    def _map_position_to_filename(self, position):
        """Map a position tuple to a patch filename.
        
        Args:
            position: (col, row) tuple
            
        Returns:
            Patch filename in the format "{col}_{row}.jpeg"
        """
        col, row = position
        return f"{col}_{row}.jpeg"

    def _rotate_90_clockwise(self, matrix):
        """Rotate a matrix 90 degrees clockwise.
        
        Args:
            matrix: 2D numpy array
            
        Returns:
            Rotated matrix
        """
        # For a 90-degree clockwise rotation:
        # 1. Transpose the matrix (swap rows and columns)
        # 2. Flip horizontally (reverse each row)
        rotated = np.transpose(matrix)
        rotated = np.fliplr(rotated)
        return rotated

    def _generate_heatmap(self, attention_scores, positions, slide_path, debug=False, normalize_scores=True):
        """Generate attention heatmap and overlay on WSI.
        
        Args:
            attention_scores: Attention scores from model
            positions: List of (col, row) positions for each patch at high magnification
            slide_path: Path to original WSI for overlay
            debug: Whether to print debug information
            normalize_scores: Whether to normalize attention scores to [0,1] range
        """
        # Print initial debug info
        if debug:
            print("\nDebug information:")
            print(f"Attention scores shape: {attention_scores.shape}")
            print(f"Number of positions: {len(positions)}")
            print(f"Normalizing scores: {normalize_scores}")
            pass
        
        # Convert attention scores to numpy and flatten
        attention_scores = attention_scores.cpu().numpy().flatten()
        if debug:
            print(f"Raw attention scores range: [{np.min(attention_scores):.4f}, {np.max(attention_scores):.4f}]")
            pass
        
        # Normalize attention scores to [0,1] range if requested
        if normalize_scores:
            attention_scores = exposure.rescale_intensity(attention_scores, out_range=(0, 1))
            if debug:
                print(f"Normalized attention scores range: [{np.min(attention_scores):.4f}, {np.max(attention_scores):.4f}]")
                pass
        
        # Verify we have matching numbers of scores and positions
        if len(attention_scores) != len(positions):
            pass
        
        # Safely get top 5 attention scores
        if debug:
            num_scores = min(5, len(attention_scores))
            top_indices = np.argsort(attention_scores)[-num_scores:][::-1]
            print(f"\nTop {num_scores} attention scores and positions:")
            for idx in top_indices:
                try:
                    pos = positions[idx]
                    filename = self._map_position_to_filename(pos)
                    print(f"Position (col,row): {pos}, Score: {attention_scores[idx]:.4f}, Filename: {filename}")
                    pass
                except IndexError as e:
                    print(f"Error accessing position {idx}: {e}")
                    print(f"Position array length: {len(positions)}")
                    print(f"Attention scores array length: {len(attention_scores)}")
                    raise ValueError("Mismatch between attention scores and positions arrays") from e
        
        # Get the maximum dimensions needed for the heatmap
        try:
            max_col = max(p[0] for p in positions) + 1
            max_row = max(p[1] for p in positions) + 1
            if debug:
                print(f"\nHeatmap dimensions: {max_row}x{max_col}")
                pass
        except ValueError as e:
            raise ValueError("Could not calculate heatmap dimensions from positions") from e
        
        # Create heatmap with correct dimensions
        heatmap = np.zeros((max_row, max_col))
        patch_mask = np.zeros((max_row, max_col), dtype=bool)
        
        # Fill in attention scores and track patch positions
        if debug:
            print("\nMapping attention scores to positions...")
            pass
        for i, (score, pos) in enumerate(zip(attention_scores, positions)):
            try:
                col, row = pos
                heatmap[row, col] = score
                patch_mask[row, col] = True
            except Exception as e:
                raise ValueError(f"Error mapping attention score to position: {e}") from e
        
        if debug:
            print(f"Number of patches mapped: {np.sum(patch_mask)}")
            print(f"Non-zero positions in heatmap: {np.count_nonzero(heatmap)}")
            print(f"Heatmap value range: [{np.min(heatmap[patch_mask]):.4f}, {np.max(heatmap):.4f}]")
            pass
        
        # Save the original heatmap before any transformations
        original_heatmap = heatmap.copy()
        
        if debug:
            print(f"Original heatmap dimensions: {original_heatmap.shape[0]}x{original_heatmap.shape[1]}")
            pass
        
        # Upscale heatmap while maintaining aspect ratio
        scale_factor = 32  # Same as training
        heatmap_upscaled = transform.resize(
            original_heatmap,
            (original_heatmap.shape[0] * scale_factor, original_heatmap.shape[1] * scale_factor),
            order=0,  # Nearest neighbor interpolation
            preserve_range=True,
            anti_aliasing=False
        )
        
        # Convert to uint8 for visualization, but handle the range appropriately
        if normalize_scores:
            # If already normalized, just scale to 0-255
            heatmap_uint8 = img_as_ubyte(heatmap_upscaled)
        else:
            # If using raw scores, normalize just for visualization
            heatmap_uint8 = img_as_ubyte(exposure.rescale_intensity(heatmap_upscaled))
        
        # Apply winter colormap using matplotlib
        winter_cmap = plt.get_cmap('winter')
        colored_heatmap = winter_cmap(heatmap_uint8 / 255.0)
        colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
        
        if slide_path:
            # Open slide and get dimensions
            slide = openslide.OpenSlide(slide_path)
            level0_dims = slide.level_dimensions[0]
            
            # Find level with appropriate resolution
            target_size = 2048
            suitable_level = 0
            for level, dims in enumerate(slide.level_dimensions):
                if max(dims) <= target_size:
                    suitable_level = level
                    break
            
            if debug:
                print(f"\nOverlay information:")
                print(f"Using slide level: {suitable_level}")
                print(f"Level dimensions: {slide.level_dimensions[suitable_level]}")
                pass
            
            # Read the WSI at suitable level
            wsi_img = slide.read_region((0,0), suitable_level, slide.level_dimensions[suitable_level])
            wsi_img = np.array(wsi_img)
            wsi_img = cv2.cvtColor(wsi_img, cv2.COLOR_RGBA2RGB)
            
            # Resize heatmap to match WSI dimensions
            colored_heatmap = cv2.resize(colored_heatmap, (wsi_img.shape[1], wsi_img.shape[0]))
            
            # Create overlay with alpha blending
            alpha = 0.3
            overlay = cv2.addWeighted(colored_heatmap, alpha, wsi_img, 1 - alpha, 0)
            
            return overlay
        
        return colored_heatmap

    def _save_top_attention_patches(self, slide_path, positions, attention_scores, output_dir, num_patches=5):
        """Save the top attention patches for inspection.
        
        Args:
            slide_path: Path to WSI file
            positions: List of (col, row) positions
            attention_scores: Attention scores
            output_dir: Directory to save patches
            num_patches: Number of top patches to save
        """
        if not output_dir:
            return
        
        # Create a subdirectory for top patches
        top_patches_dir = os.path.join(output_dir, 'top_patches')
        os.makedirs(top_patches_dir, exist_ok=True)
        
        # Get the top indices
        num_patches = min(num_patches, len(attention_scores))
        top_indices = np.argsort(attention_scores.cpu().numpy())[-num_patches:][::-1]
        
        # Open the slide
        slide = openslide.OpenSlide(slide_path)
        
        # Get objective power from slide metadata
        objective_power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if objective_power is None:
            objective_power = 40.0
        else:
            objective_power = float(objective_power)
        
        # Calculate first level based on objective power and base mag
        first_level = int(math.log2(objective_power/self.base_mag))
        
        # Create deep zoom generator
        dz = DeepZoomGenerator(slide, self.tile_size, 0)
        
        # Calculate levels exactly like in _extract_patches
        total_levels = len(dz.level_tiles)
        level_high = total_levels - 1 - (first_level + self.magnifications[0])
        
        # Save each top patch
        for i, idx in enumerate(top_indices):
            try:
                pos = positions[idx]
                col, row = pos
                score = attention_scores[idx].item()
                
                # Get the tile
                tile = dz.get_tile(level_high, (col, row))
                
                # Save the tile with attention score in the filename
                filename = f"rank{i+1}_score{score:.4f}_pos{col}_{row}.png"
                filepath = os.path.join(top_patches_dir, filename)
                tile.save(filepath)
                
                # print(f"Saved top patch {i+1}: {filepath}")
            except Exception as e:
                # print(f"Error saving patch at position {pos}: {e}")
                pass
        
        # print(f"Top {num_patches} patches saved to {top_patches_dir}")

    def _save_debug_heatmaps(self, attention_scores, positions, output_dir, normalize_scores=True, debug=False):
        """Save original and rotated heatmaps for comparison."""
        if not output_dir:
            return
        
        # Create debug directory
        debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Get the maximum dimensions needed for the heatmap
        max_col = max(p[0] for p in positions) + 1
        max_row = max(p[1] for p in positions) + 1
        
        # Create original heatmap
        original_heatmap = np.zeros((max_row, max_col))
        
        # Convert attention scores to numpy and handle normalization
        attention_scores_np = attention_scores.cpu().numpy().flatten()
        if normalize_scores:
            attention_scores_np = exposure.rescale_intensity(attention_scores_np, out_range=(0, 1))
        
        if debug:
            print("\nDebug heatmap statistics:")
            print(f"Raw score range: [{np.min(attention_scores_np):.4f}, {np.max(attention_scores_np):.4f}]")
        
        # Fill in attention scores
        for score, pos in zip(attention_scores_np, positions):
            col, row = pos
            original_heatmap[row, col] = score
        
        # Save original heatmap as CSV for inspection
        np.savetxt(os.path.join(debug_dir, 'original_heatmap.csv'), original_heatmap, delimiter=',', fmt='%.4f')
        
        # Create rotated heatmap
        rotated_heatmap = self._rotate_90_clockwise(original_heatmap)
        
        # Save rotated heatmap as CSV for inspection
        np.savetxt(os.path.join(debug_dir, 'rotated_heatmap.csv'), rotated_heatmap, delimiter=',', fmt='%.4f')
        
        # Save positions as CSV for inspection
        with open(os.path.join(debug_dir, 'positions.csv'), 'w') as f:
            f.write('index,col,row,score\n')
            for i, (pos, score) in enumerate(zip(positions, attention_scores_np)):
                col, row = pos
                f.write(f'{i},{col},{row},{score:.4f}\n')
        
        if debug:
            print(f"Debug files saved to: {debug_dir}")
            print(f"Original heatmap dimensions: {original_heatmap.shape[0]}x{original_heatmap.shape[1]}")
            print(f"Rotated heatmap dimensions: {rotated_heatmap.shape[0]}x{rotated_heatmap.shape[1]}")

    def process_slide(self, slide_path, output_dir=None, normalize_scores=True, debug=False):
        """Process a single WSI and return prediction with optional heatmap.
        
        Args:
            slide_path: Path to WSI file
            output_dir: Optional directory to save attention heatmap
            normalize_scores: Whether to normalize attention scores to [0,1] range
            debug: Whether to print debug information
            
        Returns:
            prediction: Binary prediction (0 or 1)
            probability: Prediction probability
            attention_map: Numpy array of attention heatmap if output_dir is provided
        """
        if debug:
            print("\n=== Starting slide processing ===")
        
        # Extract patches
        patches_high, patches_low, positions = self._extract_patches(slide_path)
        if not patches_high:
            raise ValueError("No valid patches found in the slide")
        
        if debug:
            print(f"\nFeature extraction:")
            print(f"Number of high mag patches: {len(patches_high)}")
            print(f"Number of low mag patches: {len(patches_low)}")
            print(f"Number of positions: {len(positions)}")
            
        # Compute features
        features_high = self._compute_features(patches_high, self.embedder_high)
        features_low = self._compute_features(patches_low, self.embedder_low)
        
        if debug:
            print(f"High mag features shape: {features_high.shape}")
            print(f"Low mag features shape: {features_low.shape}")
        
        # Combine features
        features = torch.cat([features_high, features_low], dim=1)
        features = features.to(self.device)
        
        if debug:
            print(f"Combined features shape: {features.shape}")
        
        # Get prediction
        with torch.no_grad():
            # The MILNet forward method returns: classes, prediction_bag, A, B
            # Where A is the attention matrix with shape [num_instances, num_classes]
            ins_prediction, bag_prediction, attention, _ = self.aggregator(features)
            
            if debug:
                print(f"\nModel outputs:")
                print(f"Instance predictions shape: {ins_prediction.shape}")
                print(f"Bag prediction shape: {bag_prediction.shape}")
                print(f"Attention matrix shape: {attention.shape}")
            
            # Get max instance prediction
            max_prediction, _ = torch.max(ins_prediction, 0)
            
            # Calculate probability based on the averaging flag
            if self.average:
                probability = (torch.sigmoid(max_prediction) + torch.sigmoid(bag_prediction)).squeeze().cpu().item() / 2.0
                if debug:
                    print("\nUsing average of instance and bag predictions")
            else:
                probability = torch.sigmoid(bag_prediction).squeeze().cpu().item()
                if debug:
                    print("\nUsing only bag prediction")
            
            # For binary classification, we want the attention for the positive class (index 0)
            # In DSMIL, attention matrix A has shape [num_instances, num_classes]
            instance_attention = attention[:, 0]
            
            if debug:
                print(f"\nPrediction details:")
                print(f"Threshold: {self.detection_threshold}")
                print(f"Raw bag prediction: {bag_prediction.item():.4f}")
                print(f"Sigmoid bag prediction: {torch.sigmoid(bag_prediction).item():.4f}")
                print(f"Raw max instance prediction: {max_prediction.item():.4f}")
                print(f"Sigmoid max instance prediction: {torch.sigmoid(max_prediction).item():.4f}")
                print(f"Final probability: {probability:.4f}")
                print(f"Classification: {'Positive' if probability > self.detection_threshold else 'Negative'}")
                print(f"\nRaw attention statistics:")
                print(f"Min attention: {instance_attention.min().item():.4f}")
                print(f"Max attention: {instance_attention.max().item():.4f}")
                print(f"Mean attention: {instance_attention.mean().item():.4f}")
        
        # Generate and save attention heatmap if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Verify attention scores before generating heatmap
            if instance_attention.shape[0] != len(positions):
                print(f"\nWarning: Attention scores count ({instance_attention.shape[0]}) does not match "
                      f"number of positions ({len(positions)})")
                return (probability > self.detection_threshold), probability, instance_attention
            
            if debug:
                # Save top attention patches for inspection
                self._save_top_attention_patches(slide_path, positions, instance_attention, output_dir)
            
            # Generate overlay with the slide (with debug info)
            overlay = self._generate_heatmap(
                instance_attention, 
                positions, 
                slide_path, 
                debug=debug,
                normalize_scores=normalize_scores
            )
            
            # Save overlay
            base_name = os.path.splitext(os.path.basename(slide_path))[0]
            overlay_path = os.path.join(output_dir, f'{base_name}_overlay.png')
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Generate raw heatmap without the slide
            raw_heatmap = self._generate_heatmap(
                instance_attention, 
                positions, 
                None, 
                debug=debug,
                normalize_scores=normalize_scores
            )
            
            # Save raw heatmap
            heatmap_path = os.path.join(output_dir, f'{base_name}_attention.png')
            cv2.imwrite(heatmap_path, cv2.cvtColor(raw_heatmap, cv2.COLOR_RGB2BGR))
            
            if debug:
                # Save debug heatmaps with proper normalization flag
                self._save_debug_heatmaps(instance_attention, positions, output_dir, normalize_scores, debug)
        
        return (probability > self.detection_threshold), probability, instance_attention

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='WSI Inference')
    parser.add_argument('--slide_path', type=str, required=True, help='Path to WSI file')
    parser.add_argument('--embedder_low', type=str, required=True, help='Path to low magnification embedder model')
    parser.add_argument('--embedder_high', type=str, required=True, help='Path to high magnification embedder model')
    parser.add_argument('--aggregator', type=str, required=True, help='Path to aggregator model')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save attention heatmap')
    parser.add_argument('--tile_size', type=int, default=224, help='Tile size')
    parser.add_argument('--background_threshold', type=float, default=15, help='Background threshold')
    parser.add_argument('--base_mag', type=float, default=20, help='Base magnification')
    parser.add_argument('--magnifications', type=int, nargs='+', default=[0, 1], help='Magnification levels')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu or cuda)')
    parser.add_argument('--detection_threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--average', action='store_true', help='Average bag and instance predictions')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--debug_model', action='store_true', help='Print detailed model debugging information')
    # parser.add_argument('--normalize_scores', action='store_true', default=True, help='Normalize attention scores to [0,1] range')
    # parser.add_argument('--raw_scores', action='store_true', help='Use raw attention scores (overrides normalize_scores)')
    parser.add_argument('--nonlinear', type=float, default=0, help='Additional nonlinear operation')
    parser.add_argument('--feature_size', type=int, default=1024, help='Size of feature vector (512 per magnification)')
    args = parser.parse_args()
    
    # # If raw_scores is specified, override normalize_scores
    # if args.raw_scores:
    #     args.normalize_scores = False
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    # print(f'Using device: {device}')
    
    # Initialize model
    model = WSIInference(
        embedder_low_path=args.embedder_low,
        embedder_high_path=args.embedder_high,
        aggregator_path=args.aggregator,
        tile_size=args.tile_size,
        background_threshold=args.background_threshold,
        base_mag=args.base_mag,
        magnifications=args.magnifications,
        detection_threshold=args.detection_threshold,
        average=args.average,
        debug_model=args.debug_model,
        nonlinear=args.nonlinear,
        feature_size=args.feature_size,
        device=device
    )
    
    # Process slide
    try:
        prediction, probability, attention = model.process_slide(
            args.slide_path, 
            args.output_dir,
            # normalize_scores=args.normalize_scores,
            debug=args.debug
        )
        print(f'\nPrediction: {"Positive" if prediction else "Negative"}')
        print(f'Probability: {probability:.4f}')
        
        # Print top attention scores if debug is enabled
        if args.debug and attention is not None:
            attention_np = attention.cpu().numpy()
            top_indices = np.argsort(attention_np)[-5:][::-1]
            print("\nTop 5 attention scores:")
            for i, idx in enumerate(top_indices):
                print(f"Rank {i+1}: Score {attention_np[idx]:.4f}")
    except Exception as e:
        print(f'Error processing slide: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 