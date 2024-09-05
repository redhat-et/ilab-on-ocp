from instructlab.sdg import generate_data
import openai
import os

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "Not_Needed"
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"]  = "http://10.88.0.1:8001/v1"
if not os.getenv("OPENAI_MODEL"):
     os.environ["OPENAI_MODEL"] = "Not_Needed"

client = openai.OpenAI()
openai.default_headers = {"Authorization": f'Bearer:{os.environ["OPENAI_API_KEY"]}'}

print(client.base_url)

generate_data(client=client,
              num_instructions_to_generate=2,
              output_dir = ".",
              taxonomy="taxonomy",
              taxonomy_base="main",
              model_name = os.environ["OPENAI_MODEL"]
              )
