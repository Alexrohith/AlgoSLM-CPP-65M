from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/cpp_tokenizer.json")

sample_code = """
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n;
    cin >> n;
    cout << n*(n+1)/2;
}
"""

encoding = tokenizer.encode(sample_code)

print("TOKENS:")
print(encoding.tokens[:50])

decoded = tokenizer.decode(encoding.ids)

print("\nDECODED:")
print(decoded)