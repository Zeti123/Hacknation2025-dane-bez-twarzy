from ollama import generate

stream = generate(
    model='SpeakLeash/bielik-7b-instruct-v0.1-gguf:Q4_K_S',
    prompt = 'oznacz w tekście wszystkie dane wrażliwe i zwróć json z listą: Nazywam się Jan Kowalski, mój PESEL to 90010112345. Mieszkam w Warszawie przy ulicy Długiej 5.',
)

print(stream['response'])

90010112345.