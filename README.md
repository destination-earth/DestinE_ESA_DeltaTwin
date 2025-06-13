# DestinE_ESA_DeltaTwin

DestinE DeltaTwin workflow creation/run/publish for Sentinel2 L2A product generation with AI

## Table of Contents

1. [Run the Processor Locally](#1---run-the-processor-locally)
   1. [Installation](#installation)
   2. [Running the Processor](#running-the-processor)
2. [Run the Processor via Delta Twin and Publish](#2---run-the-processor-via-delta-twin-and-publish)
   1. [Set up CDSE Credentials](#set-up-cdse-credentials)
   2. [Test the Delta Twin Locally](#test-the-delta-twin-locally)
   3. [Publish the Delta Twin](#publish-the-delta-twin)
3. [Output Example](#output-example)
4. [Repository Structure](#repository-structure)

## 1 - Run the Processor Locally

### Installation

To run the processor locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/destination-earth/DestinE_ESA_DeltaTwin
   cd DestinE_ESA_DeltaTwin
   ```

2. **Create and Activate a Conda Environment**:

   ```bash
   conda create -n ai_processor python==3.12.2
   conda activate ai_processor
   ```

3. **Install the Required Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Processor

To run the processor, execute the following command:

```bash
cd ai-sen2cor-processor/models/src/
python main.py your_cdse_key your_cdse_secret
```

## 2 - Run the Processor via Delta Twin and Publish

### Set up CDSE Credentials

To set up your CDSE credentials, edit the `ai-sen2cor-processor/inputs_file.json` JSON file by adding your CDSE key and secret:

```json
{
  "cdse_key": {
    "type": "string",
    "value": "your_cdse_key"
  },
  "cdse_secret": {
    "type": "string",
    "value": "your_cdse_secret"
  }
}
```

### Test the Delta Twin Locally

To test the Delta Twin locally, run the following command:

```bash
deltatwin run start_local -i inputs_file.json
```

### Publish the Delta Twin

1. **Login to DeltaTwin**:

   ```bash
   deltatwin login
   ```

2. **Publish the Component**:

   ```bash
   deltatwin component publish -t AiSen2Cor -v private 0.1
   ```

## Workflow Output Example

Below is an example of the Delta Twin output. The L1C product download, preprocessed, ingested by the model to generate the L2A product from the same band. The worflow should output the following band:  B02, B03 and B04.
![Image Alt Text](assets/TCI.svg)
![Image Alt Text](assets/B02.svg)
![Image Alt Text](assets/benchmark_results.svg)

## Repository Structure

The repository is structured as follows:

```bash
├── LICENSE
├── README.md
├── ai-sen2cor-processor
│   ├── inputs_file.json
│   ├── manifest.json
│   ├── models
│   │   └── src
│   │       │   └── auth.py
│   │       ├── cfg
│   │       │   ├── config.yaml
│   │       │   └── query_config.yaml
│   │       ├── main.py
│   │       ├── model_zoo
│   │       │   └── models.py
│   │       ├── utils
│   │       │   ├── stac_client.py
│   │       │   ├── torch.py
│   │       │   └── utils.py
│   │       └── weight
│   │           └── AiSen2Cor_EfficientNet_b2.pth
│   └── workflow.yml
├── assets
│   ├── asset_TCI.png
│   ├── asset_b02.png
│   └── asset_benchmark_results.png
└── requirements.txt
```

This structure includes the main directories and files necessary for the DestinE DeltaTwin workflow creation for Sentinel2 L2A product generation with AI.