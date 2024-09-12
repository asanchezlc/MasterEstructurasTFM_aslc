<font size="5"> **TFM - Máster Estructuras - UGR** </font>

---

**Author:**  
Antonio Sánchez López-Cuervo  
[antonioluis@ugr.es](mailto:antonioluis@ugr.es)


**Master Thesis supervised by:**  
Prof. Enrique García Macías  
[enriquegm@ugr.es](mailto:enriquegm@ugr.es)  


Prof. Juan Chiachío Ruano  
[jchiachio@ugr.es](mailto:jchiachio@ugr.es)


---

<!-- TOC tocDepth:2..3 chapterDepth:2..6 -->

- [1. Description](#1-description)
- [2. Prerequisites](#2-prerequisites)
- [3. Installation](#3-installation)
- [4. Directory structure](#4-directory-structure)
- [5. Establish paths: Steps for a New User to Set Up](#5-establish-paths-steps-for-a-new-user-to-set-up)
- [6. SAP2000 considerations](#6-sap2000-considerations)
  - [6.1. OAPI Documentation](#61-oapi-documentation)
  - [6.2. SAP2000 Model](#62-sap2000-model)
  - [6.3. Input data](#63-input-data)

<!-- /TOC -->

---

## 1. Description

This repository contains main codes used in the TFM


## 2. Prerequisites

Make sure you have the following prerequisites configured:

- Python 3.x (tested on Python 3.9.7):
- Packages included in requirements.txt

In order to facilitate the use in a different pc, creating a virtual environment is recommended

## 3. Installation

**Clone the repository locally:**
``` bash
git clone https://github.com/asanchezlc/TFM_codes.git
```

**Create virtual environment**: go to the project directory to create the virtual environment 
``` bash
python -m venv <name_of_virtual_environment>  # e.g. python -m venv .venv
```

**Activate virtual environment**
``` bash
# Windows
\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

**Upgrade pip and install requirements**
``` bash
python -m pip install --upgrade pip
pip install requirements.txt
```

## 4. Directory structure

- doc: contains documentation
- src: contains main codes:
  - `BI_1_LAUNCHER.py`: Launches and manages the Bayesian Inference.
  - `BI_2_function.py`: Contains functions used by `BI_1_LAUNCHER.py`.
  - `EfI_1_steelframe_testbeam.py`: EfI algorithm for the steel frame and for a beam toy structure.
  - `EfI_2_footbridge.py`: EfI algorithm for the footbridge.
  - `OMA_1_generate_timehistory.py`: Generates time vibration strain time history data for the footbridge.
- data: contains strain time data used in OMA.


## 5. Establish paths: Steps for a New User to Set Up

1. **Set Up the Environment Variables**:
   - You need to create a `.env` file where you will specify your username.
   - Copy the provided `env.dist` file and rename it to `.env`.
   - Open the `.env` file and replace the placeholder with your actual username. For example:
     ```plaintext
     USERNAME=asanchezlc
     ```

2. **Create the CSV File**:
   - Create a CSV file named after your username (for example, `asanchezlc.csv`).
   - The CSV should contain the necessary paths used by the script. - Ensure this file is stored in the expected directory (`src//paths`) as referenced by the code.
   
3. **CSV Content**:
   - Populate the CSV file with the paths the script will use. The paths can be directories, file paths, or any data relevant to the analysis.
   - Each path should be in its own row in the CSV. For example:
     ```csv
     /path/to/data1
     /path/to/data2
     /path/to/output
     ```

4. **Running the Script**:
   - When the script runs, it will automatically read the username from the `.env` file and locate the corresponding CSV (e.g., `asanchezlc.csv`) to fetch the paths.


## 6. SAP2000 considerations

- The program is tested with SAP2000 v23
- The model used is stored in sap2000 folders

### 6.1. OAPI Documentation

For using the OAPI, see CSI_OAPI_Documentation.chm file which is commonly on the same folder as the .exe file (in my case, C:\Program Files\Computers and Structures\SAP2000 23)

### 6.2. SAP2000 Model
Within sap2000 folder, a .sdb file has been left. Following groups must be defined in that file:
- allframes: contains all frames of the model
- allpoints: contains all points of the model
- modeshape_frames: contains frames to be used to retrieve strain mode shapes
- modeshape_points: contains points to be used to retrieve displacement mode shapes
- releaseframes_1, releaseframes_2...: contain frames in whiches partial fixity will be applied

Additionally, a dictionary with channel coordinates must be defined (channel_coordinates.json dict)

### 6.3. Input data
Input data is defined in a json (input_data.json), specifying material properties, spring characteristics and partial fixity & releases