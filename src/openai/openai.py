from openai import OpenAI
import sys
from tqdm import tqdm


def openai_api(prompt: str, task: str, api_key: str=None) -> str:
    if api_key:
        client = OpenAI(api_key)
    else:
        client = OpenAI()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": task,
            }
        ],
        model="gpt-3.5-turbo-1106",
    )

    return chat_completion.choices[0].message.content


def print_duration(task: str, time: str):
    duration = time.split(',')
    min_duration = duration[0].strip()
    max_duration = duration[1].strip()
    print(task + ", " + min_duration + ", " + max_duration)


def main(tasks: list[str]) -> int:
    prompt = """You are going to be given strings and assume this is the prompt for each one: 
                Estimate the minimum and maximum minutes (double (i.e. 30 seconds = 0.5)) it would take to complete the task 
                described in the following string. *end of prompt*

                Now If uncertain, make your best judgment. Please provide the values in CSV 
                format (min, max)"""

    for task in tqdm(tasks):
        task = task.strip()

        answer = openai_api(prompt, task)

        print_duration(task, answer)

    return 0


if __name__ == "__main__":
    sys.stdin = open("IO/input.txt", 'r')
    sys.stdout = open("IO/output.txt", 'w')

    istream = []

    for line in sys.stdin:
        istream.append(line)

    print("Task,Min_Duration,Max_Duration")
    main(istream[1:])
