# Customer Support Ticket Classification System (classify_sys.ai)

## Overview

The **Customer Support Ticket Classification System** is an advanced tool designed to automate and enhance the classification of customer support tickets. This system leverages state-of-the-art AI models to categorize tickets, assess urgency, evaluate customer sentiment, extract key information, and provide confidence scores. This structured approach aims to improve response times, customer satisfaction, and operational efficiency for support teams.

## Features

- **Categorization**: Automatically classifies tickets into predefined categories such as order issues, account access, product inquiries, and more.
- **Urgency Assessment**: Evaluates the urgency of each ticket, ranging from low to critical.
- **Sentiment Analysis**: Determines the sentiment of the customer, whether angry, frustrated, neutral, or satisfied.
- **Key Information Extraction**: Extracts crucial details from the ticket to assist support agents in quickly understanding and resolving issues.
- **Confidence Scoring**: Provides a confidence score indicating the model's certainty about its classification, helping to flag uncertain cases for human review.

## Usage

- Define your support tickets as strings.
- Use the classify_ticket function to get structured classification data.

## Installation

### Prerequisites

- Python 3.7 or higher
- Virtual environment manager (`venv`)

### Setup

Follow these steps to set up the project environment and install dependencies:

1. **Clone the Repository**:

    ```bash
   git clone https://github.com/prdepyadv/classify_sys.ai.git
   ```

2. **Navigate to the Project Directory**:

    ```bash
    python -m venv venv
    ```

3. **Create a Virtual Environment**:

    ```bash
    . .\venv\Scripts\Activate.ps1 #(For Windows)
        or
    . venv/bin/activate  #(For Linux or MacOS)
    ```

4. **Install Project Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

5. **Generate a Secret Token (Optional)**:

    ```bash
    python -c 'import secrets; print(secrets.token_hex())'
    ```

6. **Copy Environment Variables**:

    ```bash
    cp -r .env-example .env
    ```

7. **Configure Environment Variables**: Edit the .env file to set up necessary environment variables as per your setup.

## Running the Application

To start the Flask application, use the following command:

```bash
flask --app app run
```

## Contributions and Customization

- Adjust the system_prompt to better fit your business context.
- Experiment with different AI models for improved performance.
- Fine-tune the models if you have specific ticket data for training.

## Disclaimer

Ensure compliance with all data privacy regulations when using AI models for customer data processing.
