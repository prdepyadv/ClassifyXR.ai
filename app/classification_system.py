import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import json
from nis import cat
import os
from re import search
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from enum import Enum
from typing import List, Dict, Any, Mapping, Literal
from ollama_instructor.ollama_instructor_client import OllamaInstructorClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import tempfile

"""
Todo:
Email Automation feature is a game-changer for businesses looking to streamline their support operations on email.
Leveraging LLMs to identify multiple intents, generating responses and even escalate to the right teams is a technological breakthrough.
Automate upto 80% of incoming email queries to drive quick, empathetic resolutions while reducing operational costs.
Reduce support operations op-ex costs by upto 60%
Reduce in ticket volume by upto 85%
Improvement in First call response FCR by upto 20% 

## Advanced email understanding
#  Powered by LLMs, Yellow.ai Email Automation comprehends complex mail written in everyday language
1. Create email classification system that understands long, unstructured mails
2. Identifies multiple intents in the email & ticket (intent and action items), recognizes urgency and user sentiment.

## Automated workflow triggering
#  Execute relevant downstream workflows automatically based on email intent identified
3. Refer knowledge base for relevant resolutions
4. Escalate to the right team
5. Transfer to agent for complex use cases

## Empathetic, personalized resolutions
#  Go beyond templatized resolutions and give your customers a truly human self-serve experience
6. Cross-refer user details with CRM for contextual resolutions.
7. Identify user sentiment for customized responses.
8. Ensure adherence to company content guidelines while replying.

## AI-powered agent assist
#  Seamless transfer to human agent depending on customer query
9. AI-powered response suggestions to human agent
10. Automated trigger for agent transfer when negative user sentiment detected
11. Human agent looped in as a fallback for specific complex queries

## Connect with existing systems
#  Provide resolutions based on customer insights like demography, location, preferences, and more
12. Seamless integration with underlying systems, including existing ticketing systems and CRM tools
13. Conversations over mail between customers and the email bot recorded in the CRM for reference
14. Access the previous mail history stored in CRM to deliver contextual responses based on past interactions
"""

"""
Todo:
## Fraud detection in tickets
1. We can add concept to check if ticket is fraud or not based on the other user PII data like IP, Country, etc.
"""


class TicketCategory(str, Enum):
    ORDER_ISSUE = "order_issue"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_INQUIRY = "product_inquiry"
    SALES_INQUIRY = "sales_inquiry"
    TECHNICAL_SUPPORT = "technical_support"
    INFRASTRUCTURE = "infrastructure"
    GENERAL_SUPPORT = "general_support"
    BILLING = "billing"
    SHIPPING = "shipping"
    FEEDBACK_SUGGESTIONS = "feedback_suggestions"
    RETURN_REFUND = "return_refund"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    HAPPY = "happy"
    CONFUSED = "confused"


class TicketImpact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TicketSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


ticket_classification_example = {
    "category": TicketCategory.ORDER_ISSUE,
    "sentiment": CustomerSentiment.FRUSTRATED,
    "impact": TicketImpact.MEDIUM,
    "severity": TicketSeverity.HIGH,
    "priority": TicketPriority.HIGH,
    "confidence": 0.95,
    "key_information": ["order number: 12345", "placed a week ago"],
    "specific_requests": ["clarity on order status", "expected delivery date"],
    "suggested_action": "Investigate the issue with the tracking page for order #12345 and provide an update to the customer regarding their order status and expected delivery date.",
}


class TicketClassificationModel(BaseModel):
    category: TicketCategory = Field(
        description="Category of the customer support ticket"
    )
    sentiment: CustomerSentiment = Field(
        description="Sentiment of the customer expressing the issue"
    )
    impact: TicketImpact = Field(
        default=TicketImpact.LOW,
        description="""
            Impact of the ticket on the business operations.
            For example:
            - **high**: If it affects a subset of service outages affecting all users, major data breaches, critical application failures, 
            - **medium**: If it affects a subset of performance issues affecting a specific department, partial service outages, functionality issues with significant but non-critical applications,
            - **low**: If it affects minor issues that do not significantly impact the business operations
        """,
    )
    severity: TicketSeverity = Field(
        default=TicketSeverity.LOW,
        description="""
            Severity of the ticket based on the potential impact on the customer or business
            For example:
            - **critical**: If it involves complete service outage, critical security vulnerabilities, data loss affecting essential operations,
            - **high**: If it involves major functionality issues, significant performance degradation, issues affecting a large number of users,
            - **medium**: If it involves a significant issue that affects the customer or business operations,
            - **low**: If it involves minor issues that do not significantly impact the customer or business operations
        """,
    )
    priority: TicketPriority = Field(
        description="""
            Priority level for handling the ticket based on customer sentiment, impact and severity.
            For example:
            - **high**: Issues that cause a complete service outage or a severe degradation impacting critical business operations. There is no workaround, and the problem requires immediate attention,
            - **medium**: Issues that significantly impact functionality or performance but have a workaround available. These are important problems that need to be resolved quickly to avoid further escalation,
            - **low**: Issues that cause minor impact or inconvenience with no immediate threat to business operations. These can be handled during normal support hours and do not require urgent attention
        """
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the classification"
    )
    key_information: List[str] = Field(
        description="List of key points extracted from the ticket"
    )
    specific_requests: List[str] = Field(
        description="List of specific requests made by the customer, such as asking for a delivery date or requesting a refund"
    )
    suggested_action: str = Field(
        description="Brief suggestion for handling the ticket, ensuring it addresses any specific requests made by the customer"
    )

    """
    Ticket: Hі, I'm getting a bit frustrated as I haven't received any updates on my order #12345, placed a week ago.
    Checked the tracking page, but it's still blank. Can you let me know what's going on? A little clarity would be great. When can I expect my order?)
    """
    model_config = ConfigDict(
        json_schema_extra={"examples": [ticket_classification_example]}
    )


class TicketResponseModel(BaseModel):
    suggested_response: str = Field(
        description= (
            "The generated response to the customer, addressing their concerns based on the ticket content, knowledge base, and any provided context. "
            "The response should be polite, empathetic, and helpful, avoiding unsupported assumptions or providing specific details unless explicitly available in the data or context."
        )
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the response"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "suggested_response": (
                        "Hi Jane,\n\n"
                        "I completely understand your frustration and apologize for the lack of updates on your order #12345. "
                        "I'd be happy to help you get some clarity.\n\n"
                        "To better assist you, could you please allow me a moment to investigate this further? "
                        "I'll do my best to provide an update on the status of your order and an estimated delivery date as soon as possible.\n\n"
                        "Thank you for your patience and understanding. I appreciate it!\n\n"
                        "Best regards,\n"
                        "Alex, Customer Support"
                    ),
                    "confidence": 0.9,
                }
            ]
        }
    )


class ClassificationSystem:
    def __init__(self):
        self.assistant_name = "Alex"
        self.assistant_title = "Customer Support"
        self.business_context = f"""
            You are an AI assistant with name {self.assistant_name} and your title is {self.assistant_title} for a large e-commerce platform's customer support team.
            Your role is to analyze incoming customer support tickets and provide structured information to help our team respond quickly and effectively.
            
            Business Context:
            - We handle thousands of tickets daily across various categories (orders, accounts, products, technical issues, sales, feedback, billing).
            - Quick and accurate classification and response is crucial for customer satisfaction and operational efficiency."""
        self.os_models = ["llama3.1", "mistral", "gemma:7b-instruct"]
        self.os_primary_model = "llama3.1"
        self.satisfactory_confidence = 0.9
        self.products_overview_document_path = "storage/knowledge-base/aws-overview.pdf"
        self.tech_guide_document_path = "storage/knowledge-base/aws-setup-guide.pdf"
        self.aws_prescriptive_document_path = "storage/knowledge-base/aws-prescriptive-guidance.pdf"
        self.knowledge_base_documents_path = "storage/knowledge-base"
        self.vectordb_name = "faiss_index_react"


    def process_model(self, model: str, messages: List[Dict[str, Any]], pydantic_model, format: Literal["", "json"] = "json", debug: bool = False) -> Dict[str, Any]:
        try:
            print(f"Processing data using model: {model}")
            os_client = OllamaInstructorClient(debug=debug)
            response = os_client.chat_completion(
                model=model,
                pydantic_model=pydantic_model,
                format=format,
                messages=messages,
                options={"temperature": 0.25},
            )
            model_response = pydantic_model.model_validate(
                response["message"]["content"]
            )
            return {
                "model_name": model,
                "model_response": response["message"]["content"],
                "confidence": model_response.confidence,
            }
        except Exception as e:
            print(f"Error while processing data with model {model}: {e}")
            return {
                "model_name": model,
                "model_response": None,
                "confidence": 0.0,
            }


    def process_ticket(
        self,
        prompt: str,
        pydantic_model,
        user_request: str | None = None,
        format: Literal["", "json"] = "json",
        debug: bool = False,
    ) -> dict:
        if not self.os_models:
            raise ValueError("No models available for processing the ticket.")

        messages = [{"role": "assistant", "content": prompt}]
        if user_request:
            messages.append({"role": "user", "content": user_request})

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_model, model, messages, pydantic_model, format, debug)
                for model in self.os_models
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        best_response = max(results, key=lambda x: x["confidence"], default=None)

        if best_response is None or best_response["model_response"] is None:
            raise ValueError("All models failed.")

        return {"model_name": best_response["model_name"], "model_response": best_response["model_response"]}
    
    
    def classify_ticket(self, ticket_text: str) -> dict:
        ticket_classification_model = TicketClassificationModel
        ticket_classification_prompt = f"""
            {self.business_context}
            
            Your tasks:
            1. Categorize the ticket into the most appropriate category.
            2. Determine the customer's sentiment.
            3. Evaluate the impact on the business (low, medium, high).
            4. Specify the severity of the issue (low, medium, high, critical).
            5. Assign a priority level for handling the ticket (low, medium, high) based on the result of the combination of customer sentiment, impact and severity.
            6. Extract key information that would be helpful for our support team, including specific customer requests (e.g., requested information like delivery dates, order status updates).
            7. Identify and emphasize any specific requests made by the customer (e.g., asking for a delivery date, requesting a refund) and list them separately under 'specific_requests'.
            8. Suggest an initial action for handling the ticket, making sure to address any specific requests made by the customer.
            9. Provide a confidence score for your classification.
            
            Remember:
            - Be objective and base your analysis solely on the information provided in the ticket.
            - If you're unsure about any aspect, reflect that in your confidence score.
            - For 'key_information', extract specific details like order numbers, product names, or account issues.
            - The 'suggested_action' should be a brief, actionable step for our support team.
            - We prioritize ticket based on customer sentiment, impact and severity.

            Analyze the following customer support ticket and provide the requested information in the specified format while adhering to the following JSON schema: {TicketClassificationModel.model_json_schema()}.
        """

        response = self.process_ticket(
            user_request=ticket_text,
            prompt=ticket_classification_prompt,
            pydantic_model=ticket_classification_model,
        )
        return {"classification_model": response['model_name'], "classification": response['model_response']}


    def customer_insights(self, email: str) -> dict:
        # Customer insights can be fetched from the customer database or other sources like CRM
        customer_insights = {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "+1-555-123-4567",
            "demography": {
                "age": 34,
                "gender": "female",
                "income_level": "middle-income",
            },
            "location": "New York, USA",
            "preferences": {
                "preferred_contact_method": "email",
                "preferred_language": "English",
                "purchase_history": ["electronics", "home appliances", "fashion"],
            },
        }
        return customer_insights


    def previous_communication_history(self, email: str) -> List[Dict[str, Any]]:
        # Previous communication history can be fetched from the database or CRM
        previous_communication_history = []
        return previous_communication_history
    

    def fetch_customer_orders(self, email: str):
        # Example function to fetch previous orders
        return ["Order 1", "Order 2", "Order 3"]


    def fetch_billing_details(self, email: str):
        # Example function to fetch billing details
        return {"last_bill": "Last month", "next_3_bills": ["Next month 1", "Next month 2", "Next month 3"]}
    

    def load_documents(self, folder_path):
        """Load documents from the specified folder path."""
        if not os.path.exists(folder_path):
            print(f"The specified folder '{folder_path}' does not exist.")
            return []
        
        documents = []
        files = os.listdir(folder_path)
        loader_mapping = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader, ".json": TextLoader}
        if files:
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in loader_mapping:
                    file_path = os.path.join(folder_path, file)
                    try:
                        loader = loader_mapping[file_ext](file_path)
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as temp_file:
                temp_file.write("Knowledge base document")
                temp_file_path = temp_file.name
            loader = TextLoader(temp_file_path)
            documents = loader.load()
            os.remove(temp_file_path)
        return documents


    def initialize_vectordb(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        chunked_documents = text_splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(
            model="llama3.1",
        )
        vectorstore = FAISS.from_documents(chunked_documents, embeddings)
        vectorstore.save_local(self.vectordb_name)
        vectordb = FAISS.load_local(self.vectordb_name, embeddings)
        return vectordb


    def fetch_knowledge_base(self, document_path: str, search_terms: List[str]) -> str:
        if not search_terms:
            print("No search terms provided.")
            return ""
        
        search_query = " ".join(search_terms)
        
        documents = self.load_documents(self.knowledge_base_documents_path)
        if not documents:
            print("No documents found in the knowledge base.")
            return ""
        print(f"Loaded {len(documents)} documents from the knowledge base.")
        
        vectordb = self.initialize_vectordb(documents)
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6})
        print("Retriever initialized with MMR search.")

        retrieved_docs = retriever.invoke(search_query)
        if not retrieved_docs:
            print("No documents retrieved from the knowledge base inside vectordb.")
            return ""
        
        print("Retrieved documents:", retrieved_docs)
        return "\n".join([doc.page_content for doc in retrieved_docs])

        # relevant_text = []
        # try:
        #     file_path = os.path.join(document_path)
        #     print(f"Fetching knowledge base content from: {file_path}")

        #     with open(file_path, "rb") as file:
        #         reader = PyPDF2.PdfReader(file)
        #         for page in reader.pages:
        #             text = page.extract_text()
        #             for term in search_terms:
        #                 if term.lower() in text.lower():
        #                     relevant_text.append(text.strip())
        #                     break
        # except Exception as e:
        #     print(f"Error fetching knowledge base content: {e}")
        #     return ""
        
        # if not relevant_text:
        #     print(f"No relevant information found in the knowledge base.")
        #     return ""

        # return "\n".join(relevant_text)
    

    def handle_ticket_category_actions(self, ticket_text: str, email: str, classification: dict) -> str:
        try:
            additional_context = ""
            category = TicketCategory(classification['category'])
            search_terms = []
            if "key_information" in classification:
                search_terms = classification["key_information"]

            if category == TicketCategory.ORDER_ISSUE:
                previous_orders = self.fetch_customer_orders(email)
                if previous_orders:
                    additional_context += f"\n            - Previous Orders: {json.dumps(previous_orders)}"

            elif category == TicketCategory.BILLING:
                billing_details = self.fetch_billing_details(email)
                if billing_details:
                    additional_context += f"\n            - Billing Details: {json.dumps(billing_details)}"
                
            elif category == TicketCategory.SALES_INQUIRY:
                sales_overview = self.fetch_knowledge_base(self.products_overview_document_path, search_terms)
                if sales_overview:
                    additional_context += f"\n            - Products Overview: {sales_overview}"

            elif category == TicketCategory.GENERAL_SUPPORT:
                general_overview = self.fetch_knowledge_base(self.products_overview_document_path, search_terms)
                if general_overview:
                    additional_context += f"\n            - Products Overview: {general_overview}"
            
            elif category == TicketCategory.PRODUCT_INQUIRY:
                product_overview = self.fetch_knowledge_base(self.products_overview_document_path, search_terms)
                if product_overview:
                    additional_context += f"\n            - Products Overview: {product_overview}"

            elif category == TicketCategory.TECHNICAL_SUPPORT:
                technical_guide = self.fetch_knowledge_base(self.tech_guide_document_path, search_terms)
                if technical_guide:
                    additional_context += f"\n            - Technical Guide: {technical_guide}"

            elif category == TicketCategory.INFRASTRUCTURE:
                overview_guide = self.fetch_knowledge_base(self.products_overview_document_path, search_terms)
                if overview_guide:
                    additional_context += f"\n            - Infrastructure Guide (Overview): {overview_guide}"

                prescriptive_guide = self.fetch_knowledge_base(self.aws_prescriptive_document_path, search_terms)
                if prescriptive_guide:
                    additional_context += f"\n            - Infrastructure Guide (Prescriptive Guidance): {prescriptive_guide}"
            
            elif category == TicketCategory.SHIPPING:
                # Example: Implement logic for shipping-related inquiries
                additional_context += f"\n            - Shipping Information: Tracking and shipping status can be found in your account."
            
            elif category == TicketCategory.FEEDBACK_SUGGESTIONS:
                # Example: Implement logic for feedback or suggestions
                additional_context += f"\n            - Feedback/Suggestions: Thank you for your feedback! We value your input and will consider it for future improvements."
            
            elif category == TicketCategory.RETURN_REFUND:
                # Example: Implement logic for return or refund inquiries
                additional_context += f"\n            - Return/Refund Information: Please refer to our return policy or contact support for assistance with returns and refunds."
            
            elif category == TicketCategory.FEATURE_REQUEST:
                # Fetch product overview document in case the feature already exists in a product
                product_overview = self.fetch_knowledge_base(self.products_overview_document_path, search_terms)
                additional_context += f"\n            - Feature Request: Thank you for your suggestion! We appreciate your ideas and will review them for future updates."
                if product_overview:
                    additional_context += f"\n            - Related Products Information: {product_overview}"
                
            elif category == TicketCategory.ACCOUNT_ACCESS:
                # Example: Implement logic for account access issues
                additional_context += f"\n            - Account Access Help: If you're having trouble accessing your account, try resetting your password or contact support for help."

            else:
                pass

            return additional_context
        except Exception as e:
            print(f"Error handling ticket category actions: {e}")
            return ""


    def classify_and_response(self, ticket_text: str, user: dict) -> dict:
        customer_insights = self.customer_insights(email=user['email'])
        if not customer_insights:
            raise ValueError("Customer insights not available for the user.")

        #ticket_classification = {"classification_model": self.os_primary_model, "classification": ticket_classification_example}
        ticket_classification = self.classify_ticket(ticket_text)
        additional_context = self.handle_ticket_category_actions(
            ticket_text,
            user['email'],
            ticket_classification["classification"]
        )

        sentiment = ticket_classification['classification']['sentiment']
        if sentiment in ["angry", "frustrated"]:
            response_to_agent = True
        else:
            response_to_agent = False

        previous_communication_history = self.previous_communication_history(email=user['email'])
        if previous_communication_history:
            additional_context += f"\n            - Previous Communication History: {json.dumps(previous_communication_history)}"

        ticket_response_model = TicketResponseModel
        ticket_response_prompt = f"""
            {self.business_context}

            Contextual Information:
            - You already have classified the customer support ticket:
            - Ticket Text: {ticket_text}
            - Classified Ticket Details: {json.dumps(ticket_classification['classification'])}
            - Customer Insights: {json.dumps(customer_insights)}{additional_context}
            
            Instructions:
            - Please generate a polite, empathetic, and helpful response to the customer, addressing their concerns.
            - **Do not provide specific information about actions taken (e.g., shipping, investigating, system issues) unless explicitly stated in the input data.**
            - **Avoid assuming the status of any processes or systems (e.g., order tracking, shipping) unless that information is clearly provided.**
            - **Do not assume the use of any specific systems or platforms (e.g., AWS, Azure, etc.) unless they are explicitly mentioned in the ticket text.**
            - If you need to mention next steps or an investigation, do so cautiously without assuming actions not confirmed in the input data.
            - Offer a clear resolution or next steps to the customer based on the information provided.
            - Provide a confidence score for your response.

            Remember:
            - Be objective and base your analysis solely on the information provided.
            - If you're unsure about any aspect, reflect that in your confidence score.
            - Please end the response with a polite sign-off, such as "Best regards" or "Thank you for your patience" with your name and your title.

            Based on this information, please generate a polite and helpful response to the customer, addressing their concerns and providing the information they have requested, while adhering to the following JSON schema: {ticket_response_model.model_json_schema()}.
        """
        print("ticket_response_prompt", ticket_response_prompt)

        response = self.process_ticket(
            user_request=ticket_text,
            prompt=ticket_response_prompt,
            pydantic_model=ticket_response_model,
        )
        return {
            **ticket_classification,
            "response": response['model_response'],
            "response_model": response['model_name'],
            "forward_to_agent": response_to_agent
        }
