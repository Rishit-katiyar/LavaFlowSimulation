# LavaFlowSimulation 🌋💥

Welcome to LavaFlowSimulation, an advanced Python project for simulating highly detailed lava flow intensity maps on planetary surfaces. This project utilizes cutting-edge image processing techniques and algorithms to generate intricate volcanic flow patterns based on input planetary surface images.

<img width="931" alt="output diaplay ss" src="https://github.com/Rishit-katiyar/LavaFlowSimulation/assets/167756997/0c094a0c-98d9-4c0d-8032-1428812c68d1">

## Overview

LavaFlowSimulation is a comprehensive tool designed to analyze complex planetary surface data and simulate realistic lava flow patterns. It employs advanced image processing algorithms to identify eruption points, construct volcanic flow paths, and calculate volcanic flow intensity for each pixel on the planet. The resulting volcanic flow intensity map provides valuable insights into the dynamics of simulated lava flows.

## Features

- Advanced eruption point detection algorithm
- Precise volcanic flow path construction
- High-fidelity volcanic flow intensity calculation
- Customizable simulation parameters for fine-tuning
- Support for ultra-high-resolution planetary surface images

## Installation

To use LavaFlowSimulation, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/Rishit-katiyar/LavaFlowSimulation.git
```

### 2. Navigate to the Project Directory

```bash
cd LavaFlowSimulation
```

### 3. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv env
```

### 4. Activate the Virtual Environment

#### On Windows

```bash
.\env\Scripts\activate
```

#### On macOS and Linux

```bash
source env/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Once you have installed the dependencies, you can run the LavaFlowSimulation script to perform lava flow simulation. Here's how to do it:

```bash
python lava_flow_simulation.py
```

This command will execute the lava flow simulation code and generate the volcanic flow intensity map based on the provided input image.

## Customization

You can customize various parameters of the lava flow simulation to suit your needs. Open the `lava_flow_simulation.py` file and modify the following parameters:

- `random_seed`: Set the random seed for reproducibility.
- `contrast`: Adjust the contrast parameter for volcanic flow intensity generation.
- `bit_depth`: Set the bit depth of the output image.
- `flow_width_factor`: Control the thickness of volcanic flow rivers (higher values result in thicker flows).
- `flow_detection_limit`: Set the limit for volcanic flow detection.

Experiment with these parameters to achieve desired lava flow simulation results.

## Advanced Usage

For advanced users, LavaFlowSimulation provides additional options and capabilities:

### Custom Input Image

You can provide a custom input image for lava flow simulation. Ensure that the image is in a compatible format (e.g., TIFF, PNG) and has sufficient resolution for accurate simulation.

```bash
python lava_flow_simulation.py --input custom_image.tif
```

### Output Directory

You can specify a custom output directory for saving the generated volcanic flow intensity map.

```bash
python lava_flow_simulation.py --output-dir /path/to/output/directory
```

### High-Performance Mode

For faster execution on multi-core processors, enable high-performance mode.

```bash
python lava_flow_simulation.py --high-performance
```

## Contributing

Contributions to LavaFlowSimulation are welcome! If you have suggestions for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
