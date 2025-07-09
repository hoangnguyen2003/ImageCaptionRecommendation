# ImageCaptionRecommendation

### Table of contents
* [Introduction](#star2-introduction)
* [Installation](#wrench-installation)
* [How to run](#zap-how-to-run) 
* [Contact](#raising_hand-questions)

## :star2: Introduction

* <p align="justify">Developed a system that recommends the most suitable captions for an image from a predefined set using CLIP.</p>
* <p align="justify">Implemented image embedding extraction and cosine similarity scoring to rank captions based on relevance.</p>

![qa](/configs/dog.jpg)
![captions](/images/best_captions.PNG)

## :wrench: Installation

<p align="justify">Step-by-step instructions to get you running ImageCaptionRecommendation:</p>

### 1) Clone this repository to your local machine:

```bash
git clone https://github.com/hoangnguyen2003/ImageCaptionRecommendation.git
```

A folder called `ImageCaptionRecommendation` should appear.

### 2) Install the required packages:

Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://www.anaconda.com/docs/getting-started/miniconda/install).

You can re-create our conda enviroment from `environment.yml` file:

```bash
cd ImageCaptionRecommendation
conda env create --file environment.yml
```

<p align="justify">Your conda should start downloading and extracting packages.</p>

### 3) Activate the environment:

Your environment should be called `ImageCaptionRecommendation`, and you can activate it now to run the scripts:

```bash
conda activate ImageCaptionRecommendation
```

## :zap: How to run
<p align="justify">Simply run:</p>

```bash
python main.py
```

To run the program with your own image and custom captions, modify the `configs/captions.py` file, place your image in the `configs` directory, and execute:

```bash
python main.py --image_name your_image.ext
```

`your_image.ext` should be replaced with your actual image filename and extension.

## :raising_hand: Questions
If you have any questions about the code, please contact Hoang Van Nguyen (hoangvnguyen2003@gmail.com) or open an issue.
