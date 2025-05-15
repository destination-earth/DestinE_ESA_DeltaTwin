# DestinE_ESA_DeltaTwin
DestinE DeltaTwin workflow creation for Sentinel2 L2A product generation with AI


## Installation

1. Clone the repository:

```bash
git clone https://github.com/destination-earth/DestinE_ESA_DeltaTwin
cd DestinE_ESA_DeltaTwin
```

2. Create and activate a conda environment:

```bash
conda create -n ai_processor python==3.12.2
conda activate ai_processor
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your CDSE credentials by creating a `.env` file in the root directory with the following content:

```bash
touch .env
```
then:

```
ACCESS_KEY_ID=username
SECRET_ACCESS_KEY=password
```

## Repository structure

```Bash
.
├── LICENSE
├── README.md
├── requirements.txt
├── sample-rotation
│   ├── README.md
│   ├── manifest.json
│   ├── models
│   │   └── rotation
│   │       ├── Dockerfile
│   │       └── rotate.py
│   └── workflow.yml
└── src
    ├── auth
    │   ├── __pycache__
    │   │   └── auth.cpython-312.pyc
    │   └── auth.py
    ├── cfg
    │   ├── config.yaml
    │   └── query_config.yaml
    ├── main.py
    ├── model_zoo
    │   ├── __pycache__
    │   │   └── models.cpython-312.pyc
    │   └── models.py
    ├── utils
    │   ├── __pycache__
    │   │   ├── stac_client.cpython-312.pyc
    │   │   ├── torch.cpython-312.pyc
    │   │   └── utils.cpython-312.pyc
    │   ├── stac_client.py
    │   ├── torch.py
    │   └── utils.py
    └── weight
        └── AiSen2Core_EfficientNet_b2.pth
```
