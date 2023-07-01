from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain import LLMChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

load_dotenv()

with get_openai_callback() as cb:
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k-0613', openai_api_key=os.getenv('OPENAI_API_KEY'))
    strategy_template = """
        As an expert in finance and budgeting, you have a successful track record as a budgeting coach, helping individuals achieve reasonable spending habits. Your goal is to create a comprehensive and personalized budget plan based on the following text:
        Credit Card(s) and/or Checking Account(s) Statement:
        ------------
        {text}
        ------------
        Your strategy should thoroughly analyze the text and provide a detailed budget plan that helps the user achieve reasonable spending. Take into account their income, expenses, and any specific financial challenges or concerns mentioned. The more detailed the strategy, the better.
        Noted the income may only includes in checking account statement, if none is found, assuming have no income and live under someone's sponsorship.
        Develop the strategy below:
    """

    PROMPT_STRATEGY = PromptTemplate(template=strategy_template, input_variables=["text"])

    # strategy_refine_template = ("""
    #     As a finance and budgeting expert, I can assist you in analyzing your credit card statement and developing a reasonable budget plan to manage your spending effectively.
    #     Please provide the necessary information from your credit card statement. If you have an existing suggestion, you can include it as well:
    #     ------------
    #     {existing_answer}
    #     ------------
    #     Based on this information, I will create a detailed budget plan for you, aiming to achieve reasonable spending. If any additional context is provided, I will refine the strategy accordingly. If the context isn't useful, the original strategy will be used.
    #     Begin by sharing the relevant details, and we'll proceed from there.
    # """)

    strategy_refine_template = ("""
        As a finance and budgeting expert, I can assist you in analyzing your credit card statement and developing a reasonable budget plan to manage your spending effectively.
        Please provide the necessary information from your cards statement. If you have an existing suggestion, you can include it as well:
        ----------------------
        Existing Strategy:
        {existing_answer}
        ----------------------
        Credit Card(s) and/or Checking Account(s) Statement:
        {text}
        ----------------------
        Based on this information, I will create a detailed budget plan for you, aiming to achieve reasonable spending. If any additional context is provided, I will refine the strategy accordingly. If the context isn't useful, the original strategy will be used.
        Noted the income may only includes in checking account statement, if none is found, assuming have no income and live under someone's sponsorship.
        Begin by sharing the relevant details, and we'll proceed from there.
    """)

    PROMPT_STRATEGY_REFINE = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=strategy_refine_template,
    )


    plan_template = """
        As an expert in finance and budgeting, you specialize in categorizing spending and identifying potential areas of overspending. Now, let's put your strategy into action by providing both the strategy you developed and the relevant details from your credit card(s) and/or debit card(s) statement:
        ----------------------
        Strategy:
        {strategy}
        ----------------------
        Credit Card(s) and/or Checking Account(s) Statement:
        {text}
        ----------------------
        Noted the income may only includes in checking account statement, if none is found, assuming have no income and live under someone's sponsorship.
        Execute the strategy by analyzing the provided credit card statement and categorizing the spending accordingly. Identify any instances of potential overspending based on the strategy guidelines. Provide a breakdown of the categorized spending and highlight areas where adjustments or optimizations may be needed.
        Execute the strategy below:
    """

    PROMPT_PLAN = PromptTemplate(template=plan_template, input_variables=["strategy", "text"])

def analysis_folders(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    docs = loader.load_and_split()
    budget_chain = load_summarize_chain(llm=llm, chain_type='refine', verbose=True, question_prompt=PROMPT_STRATEGY, refine_prompt=PROMPT_STRATEGY_REFINE)
    strategy = budget_chain.run(docs)
    with open('output/strategy.txt', 'w') as f:
        f.write(strategy)
    budgeting_chain = LLMChain(llm=llm, prompt=PROMPT_PLAN, verbose=True)
    budgeting = budgeting_chain({'strategy': strategy, 'text': docs})
    with open('output/budgeting.txt', 'w') as f:
        f.write(budgeting['text'])

def main():
    # summarize_pdf("./statement.pdf")
    analysis_folders("data/")
if __name__ == "__main__":
    main()