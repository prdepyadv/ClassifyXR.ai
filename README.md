# Customer Support Ticket Classification and Response System (classify_sys.ai)

## Overview

The **Customer Support Ticket Classification and Response System** is a sophisticated tool designed to automate and elevate the customer service experience. This system integrates advanced AI models and Retrieval-Augmented Generation (RAG) techniques to classify tickets, assess urgency, evaluate customer sentiment, and extract key information. It retrieves relevant knowledge base documents to inform and generate contextually accurate, empathetic responses. By leveraging multi-model AI integration, parallel processing, and comprehensive knowledge base utilization, this system significantly improves response times, customer satisfaction, and operational efficiency for support teams.

## Features

- **Multi-Model AI Integration**: Leverages multiple open-source models, including LLaMA 3.1, Mistral, and Gemma:7B-Instruct, to process and classify customer support tickets. The system selects the most accurate response based on a confidence score.
- **Parallel Processing**: Implements parallel processing using `ThreadPoolExecutor`, allowing the system to run multiple models concurrently. This reduces response time while maintaining high-quality, accurate results.
- **Knowledge Base Integration**: Supports loading and processing of documents from a knowledge base, including PDFs, DOCX, TXT, and JSON files. Utilizes FAISS vector databases for fast and precise retrieval of relevant information based on customer inquiries.
- **Retrieval-Augmented Generation (RAG)**: Enhances response generation by retrieving the most relevant documents from the knowledge base using Maximum Marginal Relevance (MMR). The retrieved information is integrated into the generative model to produce contextually accurate and informative responses.
- **Context-Aware Responses**: Generates responses that are enriched with customer insights, previous communication history, and specific information retrieved from the knowledge base. Ensures responses are empathetic, relevant, and aligned with customer needs.
- **Customizable Business Context**: Adapts responses to specific business contexts, dynamically integrating relevant business information into the response generation. Supports various ticket categories such as orders, accounts, technical issues, billing, and more.
- **Sentiment and Urgency Detection**: Automatically detects the sentiment (e.g., frustrated, angry) and urgency of tickets, prioritizing high-impact tickets for manual review by support agents when necessary.
- **Structured Responses**: Ensures responses are structured according to predefined JSON schemas, providing clear, actionable insights for customer support teams.
- **Feature Request Handling**: Manages feature requests by retrieving relevant product information from the knowledge base, guiding customers to existing features that might meet their needs.
- **Customer Insights Integration**: Incorporates customer insights, including demographics, preferences, and location, into the ticket classification and response process, personalizing the support experience.
- **Previous Communication History Utilization**: Retrieves and integrates previous communication history to provide contextually accurate and consistent responses, improving the overall customer support experience.

## Usage

- Define your support tickets as strings.
- Use the `classify_and_response` function to get structured classification data and generate contextually enriched responses.
- The system will automatically retrieve relevant documents from the knowledge base to inform the response generation.

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

- Adjust the business_context and system_prompt to better fit your business context.
- Experiment with different AI models or embeddings for improved performance.
- Customize the vector database (e.g., FAISS) settings to optimize retrieval based on your knowledge base.
- Fine-tune the models if you have specific ticket data for training to improve classification and response accuracy.

## Disclaimer

Ensure compliance with all data privacy regulations when using AI models for customer data processing.
