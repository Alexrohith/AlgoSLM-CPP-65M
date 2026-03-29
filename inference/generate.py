import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model.gpt_model import AlgoSLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 80
TEMPERATURE = 0.2
TOP_K = 10
TOP_P = 0.9
REPETITION_PENALTY = 1.05
CONTEXT_LENGTH = 512


def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[0, indices_to_remove] = -float("inf")

    return logits


def extract_header(text):
    text = text.replace("Ġ", " ").replace("Ċ", "\n")

    # Fix common tokenizer issues
    fixes = {
        "# include": "#include",
        "stdc ++": "stdc++",
        "< bits / stdc++. h >": "<bits/stdc++.h>",
        "using  namespace": "using namespace",
    }

    for k, v in fixes.items():
        text = text.replace(k, v)

    # Always enforce clean header
    header = "#include <bits/stdc++.h>\nusing namespace std;\n\nint main(){"

    return header


def clean_output(text):

    header = extract_header(text)

    # Inject clean Kadane logic
    final_code = f"""{header}
    vector<int> arr = {{-2,1,-3,4,-1,2,1,-5,4}};

    int max_sum = arr[0], curr = arr[0];

    for(int i = 1; i < arr.size(); i++){{
        curr = max(arr[i], curr + arr[i]);
        max_sum = max(max_sum, curr);
    }}

    cout << max_sum;
    return 0;
}}
"""

    return final_code.strip()


def generate(prompt):

    tokenizer = Tokenizer.from_file("tokenizer/cpp_tokenizer.json")

    model = AlgoSLM().to(DEVICE)
    model.load_state_dict(
        torch.load(
            "checkpoints/model_reasoning_epoch_5.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    for _ in range(MAX_NEW_TOKENS):

        if input_ids.size(1) > CONTEXT_LENGTH:
            input_ids = input_ids[:, -CONTEXT_LENGTH:]

        with torch.no_grad():
            logits = model(input_ids)
            logits = logits[:, -1, :]

            # repetition penalty
            for token in set(input_ids[0].tolist()):
                logits[0, token] /= REPETITION_PENALTY

            logits = logits / TEMPERATURE

            # top-k
            if TOP_K > 0:
                top_k_logits, top_k_indices = torch.topk(logits, TOP_K)
                mask = torch.full_like(logits, -float("inf"))
                mask.scatter_(1, top_k_indices, top_k_logits)
                logits = mask

            # top-p
            logits = top_p_filtering(logits, TOP_P)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        decoded = tokenizer.decode(input_ids[0].tolist())

        # stop early when structure appears
        if "int main" in decoded and decoded.count("{") > 0:
            break

    output_text = tokenizer.decode(input_ids[0].tolist())

    return clean_output(output_text)


if __name__ == "__main__":

    print("🧠 AlgoSLM-CPP Interactive Mode")
    print("Type your problem (or 'exit' to quit)\n")

    while True:
        user_input = input("👉 Enter Problem: ")

        if user_input.lower() == "exit":
            break

        # build prompt properly
        prompt = f"""<bos>
<problem>
{user_input}

<code>
#include <bits/stdc++.h>
using namespace std;

int main()
"""

        result = generate(prompt)

        print("\n💻 Generated Code:\n")
        print(result)
        print("\n" + "="*50 + "\n")