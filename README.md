# AIT-NLP-Assignment3

This project is a web application that translates from English to Hindi. The frontend is built with React, while the backend is powered by Flask. The model predicts the next words that might follow the input phrase and returns them to be displayed on the website.

<hr>

## 🎥 **App Demo**

![NLP_A3-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/84151986-ba57-45ee-b292-cdc9e3d9df79)

![image](https://github.com/user-attachments/assets/c5348661-3b3c-4321-8aa0-3282cc9af5c7)
_Screenshot of the Translation Web App Interface_


<hr>

## 🚀 **Features**

- 🖥️ **Frontend:** A React-based UI where users enter a phrase and specify the sequence length for text generation.<br>

- 🧠 **Backend:** A Flask-based API that processes the input, loads an LSTM language model, and generates text.<br>

- 📖 **Model:** Pre-trained models, along with English and Hindi Tokenizers.<br>

<hr>

The structure is organized as follows:

```
AIT-NLP-Assignment3/
│── app/
│   ├── client/   # React frontend
│   │   ├── Dockerfile
│   │   ├── public/
│   │   ├── src/
│   │
│   ├── server/   # Flask backend
│   │   ├── models/   # Stores trained model, tokenizers
│   │   ├── app.py
│   │   ├── utils.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   ├── docker-compose.yml  # Docker Compose configuration
│
│
│── notebooks/
│   ├── loss_plots  # loss plot imgs
│   ├── myfile.ipynb
│
│── README.md
```

<hr>

## 🛠️ How It Works

### Frontend (React)

- The user enters:
  - A phrase (prompt) to translate.
- The input is sent as query parameters to the Flask backend (/translate endpoint).
- The translation is displayed on the website.

### Backend (Flask)

- The Flask server receives the request at /translate with:
  - prompt → The starting phrase for text generation.
- The server loads:
  - model-additive.pkl
  - src_tokenizer.pkl (English Tokenizer)
  - trg_tokenizer.pkl (Hindi Tokenizer)
- The translate function:
  - Passes the prompt to the translate_sentence function along with the model, src_tokenizer and trg_tokenizer.
  - The translate_sentence function then:
    - Prepares Input: The source sentence is tokenized and encoded, then moved to the specified device (CPU/GPU).
    - Initializes Translation: Starts the target sequence with a start-of-sequence (<sos>) token.
    - Generates Translation: In a loop, the model predicts the next token, appends it to the target sequence, and stops if an end-of-sequence (<eos>) token is predicted or the max length is reached.
    - Decodes Output: Converts the predicted token sequence back into a human-readable sentence.
    - Returns Translation
- The generated translation is returned as JSON to the frontend.

<hr>

### Application Endpoints

- **Frontend (React app):** Runs on http://localhost:3000
- **Backend (Flask API):** Runs on http://localhost:5000

### API Endpoints

#### **`GET /`**- Returns author information.

#### **`GET /translate`** - Takes a prompt passes it to the model for translation, and returns the result.

- Description: Generates translation based on a user-provided prompt.
- Parameters:
  - prompt (string) → Text to translate.
- Example Request:
  ```
  curl "http://localhost:5000/translate?prompt=you"
  ```
- Response Format:
  ```
  {
      "translation": "आप"
  }
  ```

## Installation and Setup

### Install NVIDIA Docker Toolkit:

Install the NVIDIA Container Toolkit, which allows Docker to interface with the GPU. Since the model is trained on gpu, the server container will crash. Or run the server and client on your device locally (refer to: Running the services separately ↓)

```
sudo apt-get update
sudo apt-get install -y nvidia-driver-<version>  # Replace with your driver version
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```

### Clone the Repository

```
git clone https://github.com/Suryansh2204/AIT-NLP-Assignment3.git
cd your-repository
```

## Setup and Running the Application

This project can be run using Docker. Follow the steps below to set up and run the application.

### Running the Application

1. Navigate to the app directory:

   ```
   cd app
   ```

2. Run Docker Compose:

   ```
   docker compose up -d
   ```

This command will build the images and start the containers in detached mode.

#### Running the services separately (Alternative)

##### Install Backend Dependencies

```
cd server
pip install -r requirements.txt
```

#### Install Frontend Dependencies

```
cd client
npm install
```

#### Run the Flask Backend

```
cd server
python app.py
```

#### Run the React Frontend

```
cd client
npm start
```

- Open http://localhost:3000/ in your browser.
