import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

# Streamlit frontend
def main():
    # Title of the app
    st.title("URL Summarizer with Groq API")

    # Input section for the Groq API key
    groq_api_key = st.text_input("Groq API Key",value="",type="password")

    # Initialize the ChatGroq LLM only after the user provides the Groq API key
    if groq_api_key:
        # Initialize the ChatGroq LLM
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

        # Define prompt template
        prompt_template = """
        Provide a summary of the following content in 300 words:
        Content:{text}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

        # Function to summarize the URL
        def summarize_url(url: str):
            # Validate URL
            if not validators.url(url):
                raise ValueError("Invalid URL")

            try:
                # Load content from the provided URL
                loader = UnstructuredURLLoader(urls=[url], ssl_verify=False,
                                                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()

                # Generate the summary
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                return output_summary

            except Exception as e:
                raise Exception(f"Error while processing the URL: {e}")

        # Input section for the URL
        url_input = st.text_input("Enter the URL you want to summarize:")

        if st.button("Summarize"):
            if url_input:
                try:
                    # Call the summarization function
                    summary = summarize_url(url_input)
                    st.write("Summary:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please enter a valid URL.")
    else:
        st.info("Please enter your Groq API key to get started.")

if __name__ == "__main__":
    main()
