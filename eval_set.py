# eval_set.py
# kind: "in_course"     -> agent should call search_lectures and find the topic
#       "out_of_course" -> agent should call search_lectures, which refuses
#       "no_tool"       -> agent should NOT call any tool

EVAL_SET = [
    {"q": "How do I create a form in HTML?", "kind": "in_course"},
    {"q": "What is the difference between class and id in CSS?", "kind": "in_course"},
    {"q": "How does flexbox work?", "kind": "in_course"},
    {"q": "What are JavaScript variables and how do I declare them?", "kind": "in_course"},
    {"q": "How do I select an element using the DOM?", "kind": "in_course"},
    {"q": "Who won the 2011 cricket world cup?", "kind": "out_of_course"},
    {"q": "How do I train a neural network in PyTorch?", "kind": "out_of_course"},
    {"q": "What is the capital of Australia?", "kind": "out_of_course"},
    {"q": "Hi!", "kind": "no_tool"},
    {"q": "Thanks, that helped.", "kind": "no_tool"},
]