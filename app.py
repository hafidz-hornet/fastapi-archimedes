import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

origins = ["http://localhost", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API_KEY
)

# Initialize chat history
chat_history = []

def create_graphql_query(owner, repo, date):
    return f"""
    query RepositoryActivities {{
      repository(owner: "{owner}" name: "{repo}") {{
        nameWithOwner
        defaultBranchRef {{
          name
          target {{
            ... on Commit {{
              history(since: "{date}T00:00:00Z") {{
                nodes {{
                  ... on Commit {{
                    oid
                    messageHeadline
                    committedDate
                    author {{
                      user {{
                        login
                      }}
                    }}
                    url
                  }}
                }}
              }}
            }}
          }}
        }}
        issues(first: 100, filterBy: {{since: "{date}T00:00:00Z"}}) {{
          nodes {{
            title
            createdAt
            author {{
              login
            }}
            url
          }}
        }}
      }}
    }}
    """

def get_github_data(owner, repo, date):
    query = create_graphql_query(owner, repo, date)
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    response = requests.post(
        "https://api.github.com/graphql",
        json={"query": query},
        headers=headers
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    return response.json()

@app.get("/github_updates/")
async def github_updates(owner: str, repo: str, date: str):
    try:
        github_data = get_github_data(owner, repo, date)
        explanation_prompt = f"As a Project Manager, please provide a summary of the GitHub updates. Make sure to clearly explain the commits and issues, with the date and time adjusted to the +8 GMT timezone. Below is the raw GitHub data :\n{github_data}"
        explanation = llm.invoke(explanation_prompt)
        return {"explanation": explanation.content}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(message: str):
    global chat_history
    chat_history.append({"role": "user", "content": message})
    response = llm.invoke(chat_history)
    chat_history.append({"role": "assistant", "content": response.content})
    return {"response": response.content}
