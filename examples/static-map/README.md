# CLSS Static Protein Domain Map

A publication-quality static visualization tool for protein domains through multi-modal embeddings using the CLSS (Contrastive Learning of Sequence and Structure) model.

## Overview

This application creates high-resolution static 2D visualizations of protein domains by:

1. **Processing Multi-Modal Data**: Reading protein sequences (FASTA) and structures (PDB) 
2. **AI-Powered Embeddings**: Using the CLSS model to generate unified embeddings for both sequences and structures
3. **Dimensionality Reduction**: Applying t-SNE to project high-dimensional embeddings to 2D space
4. **Static Visualization**: Creating publication-quality scatter plots using Matplotlib/Seaborn

## Features

- 📊 **Multi-modal visualization** - sequences and structures in the same embedding space
- 🎨 **Custom color mapping** - use hex colors with custom legend from CSV file
- 🔍 **Custom marker shapes** - specify marker shapes per data point (circle, square, diamond, star, etc.)
- 🖌️ **Advanced marker styling** - customize size, transparency, and borders per marker
- 📍 **Smart legend placement** - legend always outside plot (right, left, top, bottom) to avoid obscuring data
- 📏 **Axis control** - set custom x/y axis limits for focused views
- 🎨 **Border customization** - control marker edge color and width
- 📐 **High DPI output** - publication-quality images (300+ DPI)
- 📄 **Multiple formats** - Export to PNG, PDF, or SVG
- 💾 **Smart caching** - avoids recomputing expensive operations
- 🎯 **Flexible sizing** - customizable figure dimensions

## Pipeline Overview

```
CSV Dataset → Load Domain Data → Load FASTA/PDB Files → CLSS Model 
    ↓
Generate Embeddings → t-SNE Reduction → Create Dataframe → Static Plot 
    ↓
Export Image (PNG/PDF/SVG)
```

## Usage

### Basic Command

```bash
python generate.py \
    --dataset-path domains.csv \
    --id-column domain_id \
    --label-column fold_class \
    --fasta-path-column fasta_file \
    --pdb-path-column pdb_file \
    --output-path output.png
```

### Full Example

```bash
python generate.py \
    --dataset-path datasets/ecod_domains.csv \
    --id-column domain_id \
    --label-column architecture \
    --fasta-path-column sequence_path \
    --pdb-path-column structure_path \
    --output-path visualization.png \
    --model-repo guyyanai/CLSS \
    --model-filename h32_r10.lckpt \
    --tsne-perplexity 50 \
    --tsne-max-iterations 1000 \
    --hex-color-column custom_color \
    --marker-size 80 \
    --marker-size-column size_values \
    --marker-shape-column marker_shape \
    --alpha 0.8 \
    --alpha-column transparency \
    --dpi 300 \
    --output-format png \
    --figsize-width 14 \
    --figsize-height 12 \
    --legend-title "Protein Architecture" \
    --legend-position right \
    --x-min -50 \
    --x-max 50 \
    --y-min -40 \
    --y-max 40 \
    --edge-color white \
    --edge-width 0.8 \
    --custom-legend-csv legend.csv \
    --use-pdb-sequences \
    --use-record-id \
    --cache-path ./cache
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset-path` | ✅ | - | Path to CSV file with domain data |
| `--id-column` | ✅ | - | Column name for domain IDs |
| `--label-column` | ✅ | - | Column name for labels (determines colors) |
| `--output-path` | ✅ | - | Path to output image file |
| `--fasta-path-column` | * | - | Column name for FASTA file paths |
| `--pdb-path-column` | * | - | Column name for PDB file paths |
| `--model-repo` | ❌ | `guyyanai/CLSS` | HuggingFace model repository |
| `--model-filename` | ❌ | `h32_r10.lckpt` | Model checkpoint filename |
| `--tsne-perplexity` | ❌ | 30 | t-SNE perplexity parameter |
| `--tsne-max-iterations` | ❌ | 1000 | Maximum t-SNE iterations |
| `--tsne-random-state` | ❌ | 0 | Random state for reproducibility |
| `--output-format` | ❌ | `png` | Output format (png, pdf, svg) |
| `--dpi` | ❌ | 300 | DPI for output image (higher = better quality) |
| `--figsize-width` | ❌ | 12 | Figure width in inches |
| `--figsize-height` | ❌ | 10 | Figure height in inches |
| `--hex-color-column` | ❌ | - | Column with hex color codes for custom colors |
| `--marker-size` | ❌ | 50 | Base marker size for scatter plot |
| `--marker-size-column` | ❌ | - | Column with numeric values for marker sizes |
| `--marker-shape-column` | ❌ | - | Column with marker shape codes (o, s, ^, v, D, *, +, x) |
| `--alpha` | ❌ | 0.7 | Default alpha/transparency value (0-1) |
| `--alpha-column` | ❌ | - | Column with opacity values (0-1) for marker transparency |
| `--legend-title` | ❌ | (label column) | Title for the legend |
| `--legend-position` | ❌ | `right` | Legend position outside plot (right, left, top, bottom) |
| `--x-min` | ❌ | - | Minimum x-axis limit (auto-calculated if not specified) |
| `--x-max` | ❌ | - | Maximum x-axis limit (auto-calculated if not specified) |
| `--y-min` | ❌ | - | Minimum y-axis limit (auto-calculated if not specified) |
| `--y-max` | ❌ | - | Maximum y-axis limit (auto-calculated if not specified) |
| `--edge-color` | ❌ | `black` | Color of marker edges/borders (use 'none' for no border) |
| `--edge-width` | ❌ | `0.5` | Width of marker edges/borders in points |
| `--custom-legend-csv` | ❌ | - | Path to CSV with custom legend (class,color columns) for hex colors |
| `--exclude-structures` | ❌ | False | Exclude structure embeddings |
| `--use-pdb-sequences` | ❌ | False | Extract sequences from PDB files |
| `--use-record-id` | ❌ | False | Use domain ID as FASTA record ID |
| `--cache-path` | ❌ | - | Directory for caching intermediate results |

\* At least one of `--fasta-path-column` or `--pdb-path-column` must be provided.

## Input Format

### CSV Dataset

The input CSV should contain one row per protein domain with the following columns:

```csv
domain_id,architecture,fasta_file,pdb_file,custom_color
d1a0pa_,beta barrel,/path/to/d1a0pa_.fasta,/path/to/d1a0pa_.pdb,#FF6B6B
d1a0sa_,alpha helix,/path/to/d1a0sa_.fasta,/path/to/d1a0sa_.pdb,#4ECDC4
```

### Required Columns
- Domain identifier column (specified by `--id-column`)
- Label/classification column (specified by `--label-column`)
- At least one data source column:
  - FASTA file paths (specified by `--fasta-path-column`), OR
  - PDB file paths (specified by `--pdb-path-column`)

### Optional Columns
- `hex_color`: Custom hex color codes (e.g., `#FF6B6B`)
- `marker_size`: Custom marker sizes (numeric values)
- `marker_shape`: Marker shape codes (e.g., `o`, `s`, `^`, `v`, `D`, `*`)
- `alpha`: Transparency values (0.0 to 1.0)

### Custom Legend CSV (for use with `--hex-color-column`)

When using custom hex colors, you can provide a legend CSV file to map colors to class names:

```csv
class,color
Alpha helix,#FF6B6B
Beta sheet,#4ECDC4
Loop region,#45B7D1
Disordered,#FFA07A
```

**Required columns:**
- `class`: Display name for the legend entry
- `color`: Hex color code (e.g., `#FF6B6B`, `#4ECDC4`)

Use with `--custom-legend-csv legend.csv` alongside `--hex-color-column`.

## Output Formats

### PNG (Default)
```bash
--output-path result.png --output-format png --dpi 300
```
Best for: General use, presentations, web display

### PDF
```bash
--output-path result.pdf --output-format pdf
```
Best for: Publications, vector graphics, high-quality printing

### SVG
```bash
--output-path result.svg --output-format svg
```
Best for: Web graphics, further editing in vector graphics software

## Marker Shapes

Supported marker shapes (for `--marker-shape-column` values):

| Code | Shape | Description |
|------|-------|-------------|
| `o` | ● | Circle (default) |
| `s` | ■ | Square |
| `^` | ▲ | Triangle up |
| `v` | ▼ | Triangle down |
| `<` | ◄ | Triangle left |
| `>` | ► | Triangle right |
| `D` | ◆ | Diamond |
| `*` | ★ | Star |
| `+` | + | Plus |
| `x` | × | X |
| `p` | ⬟ | Pentagon |
| `h` | ⬡ | Hexagon |

## Examples

### Minimal Example (Sequences Only)
```bash
python generate.py \
    --dataset-path data/domains.csv \
    --id-column domain_id \
    --label-column fold \
    --fasta-path-column fasta_path \
    --output-path fold_map.png
```

### High-Quality PDF for Publication
```bash
python generate.py \
    --dataset-path data/domains.csv \
    --id-column domain_id \
    --label-column architecture \
    --fasta-path-column fasta_path \
    --pdb-path-column pdb_path \
    --output-path figure_1.pdf \
    --output-format pdf \
    --dpi 600 \
    --figsize-width 16 \
    --figsize-height 14 \
    --legend-title "Protein Architecture" \
    --legend-position left \
    --edge-color black \
    --edge-width 0.3
```

### Custom Colors and Shapes
```bash
python generate.py \
    --dataset-path data/domains.csv \
    --id-column domain_id \
    --label-column class \
    --fasta-path-column fasta_path \
    --hex-color-column color_hex \
    --marker-shape-column shape \
    --marker-size 100 \
    --alpha 0.9 \
    --edge-color none \
    --output-path custom_map.png
```

### Custom Legend with Hex Colors
```bash
# First, create a legend CSV file
cat > legend.csv << EOF
class,color
Alpha helix,#FF6B6B
Beta sheet,#4ECDC4
Loop,#45B7D1
Disordered,#FFA07A
EOF

# Generate plot with custom legend
python generate.py \
    --dataset-path data/domains.csv \
    --id-column domain_id \
    --label-column structure_type \
    --fasta-path-column fasta_path \
    --hex-color-column structure_color \
    --custom-legend-csv legend.csv \
    --legend-position right \
    --legend-title "Secondary Structure" \
    --output-path structure_map.png
```

### Focused View with Axis Limits
```bash
python generate.py \
    --dataset-path data/domains.csv \
    --id-column domain_id \
    --label-column fold \
    --fasta-path-column fasta_path \
    --x-min -30 \
    --x-max 30 \
    --y-min -25 \
    --y-max 25 \
    --output-path focused_map.png
```

### With Caching for Fast Iteration
```bash
# First run (slow - computes embeddings)
python generate.py \
    --dataset-path data/domains.csv \
    --id-column domain_id \
    --label-column fold \
    --fasta-path-column fasta_path \
    --cache-path ./cache \
    --output-path map_v1.png

# Second run (fast - uses cached embeddings)
python generate.py \
    --dataset-path data/domains.csv \
    --id-column domain_id \
    --label-column fold \
    --fasta-path-column fasta_path \
    --cache-path ./cache \
    --marker-size 80 \
    --alpha 0.9 \
    --output-path map_v2.png
```

## Dependencies

This module requires:
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualization
- `scikit-learn` - For t-SNE dimensionality reduction
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `torch` - PyTorch for model inference
- `clss` - CLSS model package
- `esm` - ESM3 for structure embeddings
- `biopython` - FASTA file parsing

## Performance Tips

1. **Use caching** (`--cache-path`) to avoid recomputing embeddings when experimenting with visual parameters
2. **Adjust t-SNE parameters** for faster iterations during development (lower perplexity/iterations)
3. **Start with PNG** for quick previews, then generate PDF/SVG for final publication
4. **Batch processing**: Process multiple datasets by writing a shell script loop

## Troubleshooting

### Out of Memory
- Reduce batch size by processing fewer domains at once
- Use a smaller perplexity value for t-SNE
- Ensure GPU memory is available for model inference

### Slow t-SNE
- Reduce `--tsne-max-iterations` (default: 1000)
- Reduce `--tsne-perplexity` (default: 30)
- Enable caching to avoid recomputing

### Missing Files
- Verify all FASTA/PDB file paths in the CSV are correct
- Use absolute paths or ensure relative paths are correct from script location

## Related Examples

- `examples/interactive-map/` - Interactive HTML visualization with click-to-highlight features
- `examples/training/` - Training the CLSS model from scratch
- `examples/inference/` - Batch inference on protein structures

## Citation

If you use this visualization tool in your research, please cite the CLSS paper:

```bibtex
@article{clss2024,
  title={CLSS: Contrastive Learning of Sequence and Structure Embeddings},
  author={...},
  journal={...},
  year={2024}
}
```
