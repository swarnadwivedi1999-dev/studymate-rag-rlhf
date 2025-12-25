from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from reward_model import compute_reward
from rlhf_loop import optimize_prompt
from feedback import collect_feedback

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = FAISS.load_local("vector_db",embeddings, allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever()

llm = Ollama(model="mistral")

# Base prompt template
base_prompt_template = """Answer ONLY using the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{question}
"""

# Initialize with base prompt
current_prompt_template = base_prompt_template
prompt = ChatPromptTemplate.from_template(current_prompt_template)

rag = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    q = input("\nAsk your question (type 'exit' to quit): ")

    if q.lower() in ["exit", "quit"]:
        print("👋 Bye!")
        break

    answer = rag.invoke(q)
    print("\nAnswer:\n", answer)
    
    # Collect feedback from user
    while True:
        try:
            rating_input = input("\nRate this answer (1-5, or 'skip' to skip feedback): ").strip()
            if rating_input.lower() == 'skip':
                break
            
            rating = int(rating_input)
            if rating < 1 or rating > 5:
                print("Please enter a rating between 1 and 5.")
                continue
            
            # Collect feedback
            collect_feedback(q, answer, rating)
            
            # Compute reward
            reward = compute_reward(rating)
            
            # Optimize prompt based on reward
            optimized_template = optimize_prompt(base_prompt_template, reward)
            
            # Update prompt and RAG chain if template changed
            if optimized_template != current_prompt_template:
                current_prompt_template = optimized_template
                prompt = ChatPromptTemplate.from_template(current_prompt_template)
                rag = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                print(f"\n✓ Prompt optimized based on your feedback (reward: {reward})")
            
            break
        except ValueError:
            print("Please enter a valid number between 1 and 5, or 'skip'.")
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            exit(0)