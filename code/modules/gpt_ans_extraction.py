from config import *


def format_request(example):
    question = example['question']
    context = example['answer']
    return f'Answer the following question using the context provided. If you can\'t find an answer in the context ' \
           f'provided then reply by only saying: {NO_ANS_TOKEN}. \n \n Question: {question} \n Context: {context} '


# Function to check if the loop has been run 3 times within the past 60 seconds
def pause_if_needed(timestamps):  # Check if we are within the API limits
    current_timestamp = time.time()
    # Remove timestamps older than 60 seconds
    loop_timestamps = [t for t in timestamps if current_timestamp - t <= 60]

    # Check if there are at least 3 loop executions within the last 60 seconds
    if len(loop_timestamps) >= RPM:
        wait_time = current_timestamp - loop_timestamps[0]
        time.sleep(wait_time)
    return loop_timestamps


# Function to query GPT and wait until the API is ready/available to provide a response
def query_gpt(msg, timestamps):
    tries = 0
    success = False
    while not success:
        try:
            timestamps = pause_if_needed(timestamps)
            timestamps.append(time.time())
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=msg,
                temperature=0  # No creativity in the answers
            )
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            print("An exception occurred:", e)
            success = False
            tries += 1
        except Exception as e:
            print('Major error', e)
            time.sleep(10)
            if tries > 20:
                success = True
                response = "error"
            else:
                success = False
                tries += 1
        else:
            success = True
    return response, timestamps
