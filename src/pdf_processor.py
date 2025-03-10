import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="../data/chroma_db")
policy_collection = chroma_client.get_or_create_collection("company_policies")

# Define Policy Sections for All PDFs
SECTIONS = {
    # Terms & Conditions
    "About the Terms": r"1\. ABOUT THE TERMS",
    "Account Registration & Termination": r"2\. ACCOUNT REGISTRATION, SUSPENSION AND TERMINATION",
    "Placing Orders & Financial Terms": r"3\. PLACING ORDERS AND FINANCIAL TERMS",
    "Use of the Platform": r"4\. USE OF THE PLATFORM",
    "Fair Usage Policy": r"5\. FAIR USAGE POLICY",
    "Accuracy & Completeness of Information": r"6\. ACCURACY AND COMPLETENESS OF INFORMATION",
    "Listing & Selling": r"7\. LISTING AND SELLING",
    "User Information & Third-Party Tools": r"8\. USER INFORMATION AND THIRD-PARTY TOOLS",
    "Intellectual Property & Infringement": r"9\. INTELLECTUAL PROPERTY (IP) AND IP INFRINGEMENT",
    "Liabilities & Disclaimers": r"10\. DISCLAIMER AND LIABILITIES",
    "Contact Company": r"11\. CONTACT COMPANY",
    "Miscellaneous & Legal Jurisdiction": r"12\. MISCELLANEOUS PROVISIONS APPLICABLE TO AGREEMENT",

    # Privacy Policy
    "Privacy Policy Overview": r"PRIVACY POLICY",
    "Applicability of the Policy": r"1\. APPLICABILITY OF THE POLICY",
    "Collection of Information": r"2\. COLLECTION OF THE INFORMATION",
    "Use of Information": r"3\. USE OF THE INFORMATION",
    "Sharing of Information": r"4\. SHARING OF THE INFORMATION",
    "Third-Party Links & Services": r"5\. THIRD PARTY LINKS AND SERVICES",
    "Security Precautions": r"8\. SECURITY PRECAUTIONS",
    "Data Retention": r"10\. DATA RETENTION",
    "Changes to the Privacy Policy": r"13\. CHANGES TO THIS PRIVACY POLICY",
    "Grievance Officer": r"14\. GRIEVANCE OFFICER",

    # Returns, Exchange & Refunds Policy
    "Returns Overview": r"RETURNS, EXCHANGE AND REFUNDS POLICY",
    "Return options": r"RETURN OPTIONS",
    "Exchange": r"EXCHANGE",
    "Refund Queries": r"REFUND QUERIES",
    "Refund Timelines": r"WHEN WILL I GET MY REFUND",
    "Instant Refunds": r"INSTANT REFUND",
    "Return Eligibility": r"COMMON GUIDELINES FOR RETURN AND EXCHANGE",

    # Cancellation Policy
    "Cancellation Overview": r"CANCELLATION POLICY",
    "User Cancellation": r"CANCELLATION BY THE USER",
    "Supplier Cancellation": r"CANCELLATION BY THE SUPPLIER",
    "Ecom Cancellation": r"CANCELLATION BY ECOM",
    "Refunds After Cancellation": r"REFUNDS AFTER ORDER CANCELLATION",
    "Refund Processing Time": r"WHEN WILL THE USER GET THE REFUND AFTER CANCELLATION OF ORDER",
    "Discount Vouchers & Offers": r"WILL THE DISCOUNT VOUCHERS OR OTHER SUCH PROMOTIONAL OFFERS BE REINSTATED",

    # Influencer Marketing Program
    "Influencer Marketing Program": r"INFLUENCER MARKETING PROGRAM",
}

def extract_sections(pdf_path):
    """Extracts policy sections, chunks them, and stores them in ChromaDB."""
    doc = fitz.open(pdf_path)
    text = ""

    # Extract full text
    for page in doc:
        text += page.get_text("text") + "\n"

    # Get the filename (without extension) to make IDs unique
    filename = os.path.splitext(os.path.basename(pdf_path))[0]

    section_texts = {}
    current_section = None

    # Splitting text into sections based on subheading patterns
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        for section, pattern in SECTIONS.items():
            if re.match(pattern, line, re.IGNORECASE):
                current_section = section
                section_texts[current_section] = ""
                break
        if current_section:
            section_texts[current_section] += line + "\n"

    # Split large sections into smaller chunks (better retrieval)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Store each section separately in ChromaDB with unique IDs
    for section, content in section_texts.items():
        if content.strip():
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                unique_id = f"{filename}_{section}_{i}"  
                embedding = embedding_model.encode(chunk).tolist()
                policy_collection.add(
                    ids=[unique_id],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{"section": section, "source": filename}]
                )
            print(f"Stored '{section}' from '{filename}' ({len(chunks)} chunks)")

def process_all_pdfs(pdf_folder):
    """Processes all policy PDFs in the given folder."""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing: {filename}")
            extract_sections(pdf_path)

if __name__ == "__main__":
    process_all_pdfs("../data/company policies/")  # Folder where PDFs are stored
