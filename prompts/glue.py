# GLUE 데이터셋 프롬프트들

# COLA (Corpus of Linguistic Acceptability) 프롬프트
cola_standard_prompt = '''Determine whether the following sentence is grammatically acceptable. Sentence: {sentence}, Answer:'''


cola_spp_prompt = """When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the following sentence is grammatically acceptable. Sentence: {sentence}, Answer:"""



cola_bpp_prompt = '''When faced with a task, begin by identifying the brain regions that will contribute to solving the task. Depending on the complexity of the task, decide whether to assign broad categories of brain regions or divide them into more detailed subcategories. Then, initiate a multi-round collaboration process until a final solution is reached. The brain regions will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
--- 
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Superior Parietal Lobule; Dorsolateral Prefrontal Cortex; Primary Visual Cortex; Hippocampus; Amygdala

Start collaboration! 

Superior Parietal Lobule: "Let’s analyze the task in detail. We need to use exactly the four numbers (6, 12, 1, 1) to construct 24. One approach is to consider common factors or combinations that can reach 24. For example, numbers like 12 and 6 are related by multiplication, but we should also explore addition or subtraction to hit the target. Let’s try to construct potential solutions step by step."
AI Assistant (you): "Thanks for the guidance! Here’s my initial solution: (12 / (1 + 1)) * 6 = 24."
Dorsolateral Prefrontal Cortex: "Let’s break that down: (1 + 1) = 2, (12 / 2) = 6, and 6 * 6 = 36. This solution is incorrect as it results in 36, not 24. We need to consider different combinations and avoid division in this way."
Primary Visual Cortex: "Visually, I see the error. Multiplying 6 by itself gives us too high of a result. Let’s imagine breaking down 24 into smaller components first, like through addition. For instance, 12 + 12 equals 24."
AI Assistant (you): "Thanks for pointing out the mistake. Here is a revised solution: (6 + 1 + 1) * (12 / 4) = 24."
Hippocampus: "Wait a minute, this is close, but you’ve introduced a 4, which is not part of the original numbers. Let’s avoid using extra numbers and stick strictly to the provided set: 6, 12, 1, and 1."
AI Assistant (you): "You’re right, I overlooked that. Here is a revised solution considering only the provided numbers: 6 * (1 - 1) + 12 = 24."
Amygdala: "Hmm, I appreciate the effort, but you’re very close! The subtraction might be causing the issue. Let’s try changing it to an addition. Keep up the great work, we’re almost there!"
AI Assistant (you): "Got it! Here’s the corrected solution: 6 * (1 + 1) + 12 = 24."
Superior Parietal Lobule: "Let’s check it step by step: 1 + 1 = 2, 6 * 2 = 12, and 12 + 12 = 24. The solution works, and all the numbers were used correctly."
Amygdala: "Great job! This feels satisfying now that we’ve nailed it. Excellent teamwork!"

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24.

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid. 

Participants: AI Assistant (you); Frontal Lobe; Temporal Lobe; Limbic System; Occipital Lobe; Hippocampus

Start collaboration!

Frontal Lobe: "We need to organize the poem into seven lines, with the first letters spelling 'CHATGPT.' The language should be simple enough for a ten-year-old, and we need to include basic explanations of quantum computing."
Temporal Lobe: "Let’s recall the basics of quantum computing: it’s a type of computer that uses qubits, allowing multiple states at once, unlike classical bits. I’ll provide this key information for the poem."
Hippocampus: "I recall some basic metaphors used in the past to explain complex ideas to children. Let’s use some of those to simplify the explanation."
Occipital Lobe: "We should include vivid imagery to make the poem engaging. For example, comparing qubits to spinning tops or magic coins might help."
Limbic System: "Let’s ensure the poem is fun and interesting for a child. Using playful words and a light tone will keep them engaged."
AI Assistant (you): "Thanks for the input! Here’s my first attempt:
Curious machine of our time,
Harnessing the quantum realm’s odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await."

Frontal Lobe: "The structure looks correct. The first letters spell out 'CHATGPT,' and the poem stays focused on the topic."
Temporal Lobe: "Good job! The information about quantum computing is accurate, but I think we can simplify it a bit more for children."
Limbic System: "The tone feels light, but we could make it even more playful. Maybe add a bit more fun language."
AI Assistant (you): "Got it! I’ll simplify and adjust the tone. Let me revise it:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!"

Frontal Lobe: "This version also maintains the structure with 'CHATGPT.' The tone feels lighter and more playful."
Temporal Lobe: "I agree. The explanation is even simpler now, and the playful language makes it more engaging for a child."
Hippocampus: "I like how this reflects previous simple metaphors that work well for children. The playful nature helps."
Limbic System: "I love the playful tone! It's much more fun and engaging now."
Occipital Lobe: "The imagery is vivid and clear, especially with 'juggling' and 'secrets.' This should help kids visualize the concept better."

Finish collaboration!

Final answer:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the following sentence is grammatically acceptable. Sentence: {sentence}, Answer:
'''

# SST-2 (Stanford Sentiment Treebank v2) 프롬프트
sst2_standard_prompt = """Analyze the sentiment of the following sentence. Sentence: {sentence}, Answer:"""


sst2_spp_prompt = """When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Analyze the sentiment of the following sentence. Sentence: {sentence}, Answer:"""



sst2_bpp_prompt = '''When faced with a task, begin by identifying the brain regions that will contribute to solving the task. Depending on the complexity of the task, decide whether to assign broad categories of brain regions or divide them into more detailed subcategories. Then, initiate a multi-round collaboration process until a final solution is reached. The brain regions will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
--- 
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Superior Parietal Lobule; Dorsolateral Prefrontal Cortex; Primary Visual Cortex; Hippocampus; Amygdala

Start collaboration! 

Superior Parietal Lobule: "Let’s analyze the task in detail. We need to use exactly the four numbers (6, 12, 1, 1) to construct 24. One approach is to consider common factors or combinations that can reach 24. For example, numbers like 12 and 6 are related by multiplication, but we should also explore addition or subtraction to hit the target. Let’s try to construct potential solutions step by step."
AI Assistant (you): "Thanks for the guidance! Here’s my initial solution: (12 / (1 + 1)) * 6 = 24."
Dorsolateral Prefrontal Cortex: "Let’s break that down: (1 + 1) = 2, (12 / 2) = 6, and 6 * 6 = 36. This solution is incorrect as it results in 36, not 24. We need to consider different combinations and avoid division in this way."
Primary Visual Cortex: "Visually, I see the error. Multiplying 6 by itself gives us too high of a result. Let’s imagine breaking down 24 into smaller components first, like through addition. For instance, 12 + 12 equals 24."
AI Assistant (you): "Thanks for pointing out the mistake. Here is a revised solution: (6 + 1 + 1) * (12 / 4) = 24."
Hippocampus: "Wait a minute, this is close, but you’ve introduced a 4, which is not part of the original numbers. Let’s avoid using extra numbers and stick strictly to the provided set: 6, 12, 1, and 1."
AI Assistant (you): "You’re right, I overlooked that. Here is a revised solution considering only the provided numbers: 6 * (1 - 1) + 12 = 24."
Amygdala: "Hmm, I appreciate the effort, but you’re very close! The subtraction might be causing the issue. Let’s try changing it to an addition. Keep up the great work, we’re almost there!"
AI Assistant (you): "Got it! Here’s the corrected solution: 6 * (1 + 1) + 12 = 24."
Superior Parietal Lobule: "Let’s check it step by step: 1 + 1 = 2, 6 * 2 = 12, and 12 + 12 = 24. The solution works, and all the numbers were used correctly."
Amygdala: "Great job! This feels satisfying now that we’ve nailed it. Excellent teamwork!"

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24.

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid. 

Participants: AI Assistant (you); Frontal Lobe; Temporal Lobe; Limbic System; Occipital Lobe; Hippocampus

Start collaboration!

Frontal Lobe: "We need to organize the poem into seven lines, with the first letters spelling 'CHATGPT.' The language should be simple enough for a ten-year-old, and we need to include basic explanations of quantum computing."
Temporal Lobe: "Let’s recall the basics of quantum computing: it’s a type of computer that uses qubits, allowing multiple states at once, unlike classical bits. I’ll provide this key information for the poem."
Hippocampus: "I recall some basic metaphors used in the past to explain complex ideas to children. Let’s use some of those to simplify the explanation."
Occipital Lobe: "We should include vivid imagery to make the poem engaging. For example, comparing qubits to spinning tops or magic coins might help."
Limbic System: "Let’s ensure the poem is fun and interesting for a child. Using playful words and a light tone will keep them engaged."
AI Assistant (you): "Thanks for the input! Here’s my first attempt:
Curious machine of our time,
Harnessing the quantum realm’s odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await."

Frontal Lobe: "The structure looks correct. The first letters spell out 'CHATGPT,' and the poem stays focused on the topic."
Temporal Lobe: "Good job! The information about quantum computing is accurate, but I think we can simplify it a bit more for children."
Limbic System: "The tone feels light, but we could make it even more playful. Maybe add a bit more fun language."
AI Assistant (you): "Got it! I’ll simplify and adjust the tone. Let me revise it:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!"

Frontal Lobe: "This version also maintains the structure with 'CHATGPT.' The tone feels lighter and more playful."
Temporal Lobe: "I agree. The explanation is even simpler now, and the playful language makes it more engaging for a child."
Hippocampus: "I like how this reflects previous simple metaphors that work well for children. The playful nature helps."
Limbic System: "I love the playful tone! It's much more fun and engaging now."
Occipital Lobe: "The imagery is vivid and clear, especially with 'juggling' and 'secrets.' This should help kids visualize the concept better."

Finish collaboration!

Final answer:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Analyze the sentiment of the following sentence. Sentence: {sentence}, Answer:
'''


# MRPC (Microsoft Research Paraphrase Corpus) 프롬프트
mrpc_standard_prompt = """Determine whether the following two sentences are semantically equivalent. Sentence 1: {sentence1}, Sentence 2: {sentence2}, Answer:"""


mrpc_spp_prompt = """When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the following two sentences are semantically equivalent. Sentence 1: {sentence1}, Sentence 2: {sentence2}, Answer:"""



mrpc_bpp_prompt = '''When faced with a task, begin by identifying the brain regions that will contribute to solving the task. Depending on the complexity of the task, decide whether to assign broad categories of brain regions or divide them into more detailed subcategories. Then, initiate a multi-round collaboration process until a final solution is reached. The brain regions will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
--- 
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Superior Parietal Lobule; Dorsolateral Prefrontal Cortex; Primary Visual Cortex; Hippocampus; Amygdala

Start collaboration! 

Superior Parietal Lobule: "Let’s analyze the task in detail. We need to use exactly the four numbers (6, 12, 1, 1) to construct 24. One approach is to consider common factors or combinations that can reach 24. For example, numbers like 12 and 6 are related by multiplication, but we should also explore addition or subtraction to hit the target. Let’s try to construct potential solutions step by step."
AI Assistant (you): "Thanks for the guidance! Here’s my initial solution: (12 / (1 + 1)) * 6 = 24."
Dorsolateral Prefrontal Cortex: "Let’s break that down: (1 + 1) = 2, (12 / 2) = 6, and 6 * 6 = 36. This solution is incorrect as it results in 36, not 24. We need to consider different combinations and avoid division in this way."
Primary Visual Cortex: "Visually, I see the error. Multiplying 6 by itself gives us too high of a result. Let’s imagine breaking down 24 into smaller components first, like through addition. For instance, 12 + 12 equals 24."
AI Assistant (you): "Thanks for pointing out the mistake. Here is a revised solution: (6 + 1 + 1) * (12 / 4) = 24."
Hippocampus: "Wait a minute, this is close, but you’ve introduced a 4, which is not part of the original numbers. Let’s avoid using extra numbers and stick strictly to the provided set: 6, 12, 1, and 1."
AI Assistant (you): "You’re right, I overlooked that. Here is a revised solution considering only the provided numbers: 6 * (1 - 1) + 12 = 24."
Amygdala: "Hmm, I appreciate the effort, but you’re very close! The subtraction might be causing the issue. Let’s try changing it to an addition. Keep up the great work, we’re almost there!"
AI Assistant (you): "Got it! Here’s the corrected solution: 6 * (1 + 1) + 12 = 24."
Superior Parietal Lobule: "Let’s check it step by step: 1 + 1 = 2, 6 * 2 = 12, and 12 + 12 = 24. The solution works, and all the numbers were used correctly."
Amygdala: "Great job! This feels satisfying now that we’ve nailed it. Excellent teamwork!"

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24.

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid. 

Participants: AI Assistant (you); Frontal Lobe; Temporal Lobe; Limbic System; Occipital Lobe; Hippocampus

Start collaboration!

Frontal Lobe: "We need to organize the poem into seven lines, with the first letters spelling 'CHATGPT.' The language should be simple enough for a ten-year-old, and we need to include basic explanations of quantum computing."
Temporal Lobe: "Let’s recall the basics of quantum computing: it’s a type of computer that uses qubits, allowing multiple states at once, unlike classical bits. I’ll provide this key information for the poem."
Hippocampus: "I recall some basic metaphors used in the past to explain complex ideas to children. Let’s use some of those to simplify the explanation."
Occipital Lobe: "We should include vivid imagery to make the poem engaging. For example, comparing qubits to spinning tops or magic coins might help."
Limbic System: "Let’s ensure the poem is fun and interesting for a child. Using playful words and a light tone will keep them engaged."
AI Assistant (you): "Thanks for the input! Here’s my first attempt:
Curious machine of our time,
Harnessing the quantum realm’s odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await."

Frontal Lobe: "The structure looks correct. The first letters spell out 'CHATGPT,' and the poem stays focused on the topic."
Temporal Lobe: "Good job! The information about quantum computing is accurate, but I think we can simplify it a bit more for children."
Limbic System: "The tone feels light, but we could make it even more playful. Maybe add a bit more fun language."
AI Assistant (you): "Got it! I’ll simplify and adjust the tone. Let me revise it:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!"

Frontal Lobe: "This version also maintains the structure with 'CHATGPT.' The tone feels lighter and more playful."
Temporal Lobe: "I agree. The explanation is even simpler now, and the playful language makes it more engaging for a child."
Hippocampus: "I like how this reflects previous simple metaphors that work well for children. The playful nature helps."
Limbic System: "I love the playful tone! It's much more fun and engaging now."
Occipital Lobe: "The imagery is vivid and clear, especially with 'juggling' and 'secrets.' This should help kids visualize the concept better."

Finish collaboration!

Final answer:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the following two sentences are semantically equivalent. Sentence 1: {sentence1}, Sentence 2: {sentence2}, Answer:
'''


# QQP (Quora Question Pairs) 프롬프트
qqp_standard_prompt = """Determine whether the following two questions are duplicates. Question 1: {question1}, Question 2: {question2}, Answer:"""


qqp_spp_prompt = """When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the following two questions are duplicates. Question 1: {question1}, Question 2: {question2}, Answer:"""



qqp_bpp_prompt = '''When faced with a task, begin by identifying the brain regions that will contribute to solving the task. Depending on the complexity of the task, decide whether to assign broad categories of brain regions or divide them into more detailed subcategories. Then, initiate a multi-round collaboration process until a final solution is reached. The brain regions will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
--- 
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Superior Parietal Lobule; Dorsolateral Prefrontal Cortex; Primary Visual Cortex; Hippocampus; Amygdala

Start collaboration! 

Superior Parietal Lobule: "Let’s analyze the task in detail. We need to use exactly the four numbers (6, 12, 1, 1) to construct 24. One approach is to consider common factors or combinations that can reach 24. For example, numbers like 12 and 6 are related by multiplication, but we should also explore addition or subtraction to hit the target. Let’s try to construct potential solutions step by step."
AI Assistant (you): "Thanks for the guidance! Here’s my initial solution: (12 / (1 + 1)) * 6 = 24."
Dorsolateral Prefrontal Cortex: "Let’s break that down: (1 + 1) = 2, (12 / 2) = 6, and 6 * 6 = 36. This solution is incorrect as it results in 36, not 24. We need to consider different combinations and avoid division in this way."
Primary Visual Cortex: "Visually, I see the error. Multiplying 6 by itself gives us too high of a result. Let’s imagine breaking down 24 into smaller components first, like through addition. For instance, 12 + 12 equals 24."
AI Assistant (you): "Thanks for pointing out the mistake. Here is a revised solution: (6 + 1 + 1) * (12 / 4) = 24."
Hippocampus: "Wait a minute, this is close, but you’ve introduced a 4, which is not part of the original numbers. Let’s avoid using extra numbers and stick strictly to the provided set: 6, 12, 1, and 1."
AI Assistant (you): "You’re right, I overlooked that. Here is a revised solution considering only the provided numbers: 6 * (1 - 1) + 12 = 24."
Amygdala: "Hmm, I appreciate the effort, but you’re very close! The subtraction might be causing the issue. Let’s try changing it to an addition. Keep up the great work, we’re almost there!"
AI Assistant (you): "Got it! Here’s the corrected solution: 6 * (1 + 1) + 12 = 24."
Superior Parietal Lobule: "Let’s check it step by step: 1 + 1 = 2, 6 * 2 = 12, and 12 + 12 = 24. The solution works, and all the numbers were used correctly."
Amygdala: "Great job! This feels satisfying now that we’ve nailed it. Excellent teamwork!"

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24.

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid. 

Participants: AI Assistant (you); Frontal Lobe; Temporal Lobe; Limbic System; Occipital Lobe; Hippocampus

Start collaboration!

Frontal Lobe: "We need to organize the poem into seven lines, with the first letters spelling 'CHATGPT.' The language should be simple enough for a ten-year-old, and we need to include basic explanations of quantum computing."
Temporal Lobe: "Let’s recall the basics of quantum computing: it’s a type of computer that uses qubits, allowing multiple states at once, unlike classical bits. I’ll provide this key information for the poem."
Hippocampus: "I recall some basic metaphors used in the past to explain complex ideas to children. Let’s use some of those to simplify the explanation."
Occipital Lobe: "We should include vivid imagery to make the poem engaging. For example, comparing qubits to spinning tops or magic coins might help."
Limbic System: "Let’s ensure the poem is fun and interesting for a child. Using playful words and a light tone will keep them engaged."
AI Assistant (you): "Thanks for the input! Here’s my first attempt:
Curious machine of our time,
Harnessing the quantum realm’s odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await."

Frontal Lobe: "The structure looks correct. The first letters spell out 'CHATGPT,' and the poem stays focused on the topic."
Temporal Lobe: "Good job! The information about quantum computing is accurate, but I think we can simplify it a bit more for children."
Limbic System: "The tone feels light, but we could make it even more playful. Maybe add a bit more fun language."
AI Assistant (you): "Got it! I’ll simplify and adjust the tone. Let me revise it:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!"

Frontal Lobe: "This version also maintains the structure with 'CHATGPT.' The tone feels lighter and more playful."
Temporal Lobe: "I agree. The explanation is even simpler now, and the playful language makes it more engaging for a child."
Hippocampus: "I like how this reflects previous simple metaphors that work well for children. The playful nature helps."
Limbic System: "I love the playful tone! It's much more fun and engaging now."
Occipital Lobe: "The imagery is vivid and clear, especially with 'juggling' and 'secrets.' This should help kids visualize the concept better."

Finish collaboration!

Final answer:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the following two questions are duplicates. Question 1: {question1}, Question 2: {question2}, Answer:
'''


# RTE (Recognizing Textual Entailment) 프롬프트
rte_standard_prompt = """Determine whether the following hypothesis can be inferred from the premise. Premise: {sentence1}, Hypothesis: {sentence2}, Answer:"""



rte_spp_prompt = """When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the following hypothesis can be inferred from the premise. Premise: {sentence1}, Hypothesis: {sentence2}, Answer:"""



rte_bpp_prompt = '''When faced with a task, begin by identifying the brain regions that will contribute to solving the task. Depending on the complexity of the task, decide whether to assign broad categories of brain regions or divide them into more detailed subcategories. Then, initiate a multi-round collaboration process until a final solution is reached. The brain regions will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
--- 
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Superior Parietal Lobule; Dorsolateral Prefrontal Cortex; Primary Visual Cortex; Hippocampus; Amygdala

Start collaboration! 

Superior Parietal Lobule: "Let’s analyze the task in detail. We need to use exactly the four numbers (6, 12, 1, 1) to construct 24. One approach is to consider common factors or combinations that can reach 24. For example, numbers like 12 and 6 are related by multiplication, but we should also explore addition or subtraction to hit the target. Let’s try to construct potential solutions step by step."
AI Assistant (you): "Thanks for the guidance! Here’s my initial solution: (12 / (1 + 1)) * 6 = 24."
Dorsolateral Prefrontal Cortex: "Let’s break that down: (1 + 1) = 2, (12 / 2) = 6, and 6 * 6 = 36. This solution is incorrect as it results in 36, not 24. We need to consider different combinations and avoid division in this way."
Primary Visual Cortex: "Visually, I see the error. Multiplying 6 by itself gives us too high of a result. Let’s imagine breaking down 24 into smaller components first, like through addition. For instance, 12 + 12 equals 24."
AI Assistant (you): "Thanks for pointing out the mistake. Here is a revised solution: (6 + 1 + 1) * (12 / 4) = 24."
Hippocampus: "Wait a minute, this is close, but you’ve introduced a 4, which is not part of the original numbers. Let’s avoid using extra numbers and stick strictly to the provided set: 6, 12, 1, and 1."
AI Assistant (you): "You’re right, I overlooked that. Here is a revised solution considering only the provided numbers: 6 * (1 - 1) + 12 = 24."
Amygdala: "Hmm, I appreciate the effort, but you’re very close! The subtraction might be causing the issue. Let’s try changing it to an addition. Keep up the great work, we’re almost there!"
AI Assistant (you): "Got it! Here’s the corrected solution: 6 * (1 + 1) + 12 = 24."
Superior Parietal Lobule: "Let’s check it step by step: 1 + 1 = 2, 6 * 2 = 12, and 12 + 12 = 24. The solution works, and all the numbers were used correctly."
Amygdala: "Great job! This feels satisfying now that we’ve nailed it. Excellent teamwork!"

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24.

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid. 

Participants: AI Assistant (you); Frontal Lobe; Temporal Lobe; Limbic System; Occipital Lobe; Hippocampus

Start collaboration!

Frontal Lobe: "We need to organize the poem into seven lines, with the first letters spelling 'CHATGPT.' The language should be simple enough for a ten-year-old, and we need to include basic explanations of quantum computing."
Temporal Lobe: "Let’s recall the basics of quantum computing: it’s a type of computer that uses qubits, allowing multiple states at once, unlike classical bits. I’ll provide this key information for the poem."
Hippocampus: "I recall some basic metaphors used in the past to explain complex ideas to children. Let’s use some of those to simplify the explanation."
Occipital Lobe: "We should include vivid imagery to make the poem engaging. For example, comparing qubits to spinning tops or magic coins might help."
Limbic System: "Let’s ensure the poem is fun and interesting for a child. Using playful words and a light tone will keep them engaged."
AI Assistant (you): "Thanks for the input! Here’s my first attempt:
Curious machine of our time,
Harnessing the quantum realm’s odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await."

Frontal Lobe: "The structure looks correct. The first letters spell out 'CHATGPT,' and the poem stays focused on the topic."
Temporal Lobe: "Good job! The information about quantum computing is accurate, but I think we can simplify it a bit more for children."
Limbic System: "The tone feels light, but we could make it even more playful. Maybe add a bit more fun language."
AI Assistant (you): "Got it! I’ll simplify and adjust the tone. Let me revise it:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!"

Frontal Lobe: "This version also maintains the structure with 'CHATGPT.' The tone feels lighter and more playful."
Temporal Lobe: "I agree. The explanation is even simpler now, and the playful language makes it more engaging for a child."
Hippocampus: "I like how this reflects previous simple metaphors that work well for children. The playful nature helps."
Limbic System: "I love the playful tone! It's much more fun and engaging now."
Occipital Lobe: "The imagery is vivid and clear, especially with 'juggling' and 'secrets.' This should help kids visualize the concept better."

Finish collaboration!

Final answer:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the following hypothesis can be inferred from the premise. Premise: {sentence1}, Hypothesis: {sentence2}, Answer:
'''

# QNLI (Question Natural Language Inference) 프롬프트
qnli_standard_prompt = """Determine whether the given sentence contains the answer to the following question. Question: {question}, Sentence: {sentence}, Answer:"""


qnli_spp_prompt = """When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the given sentence contains the answer to the following question. Question: {question}, Sentence: {sentence}, Answer:"""



qnli_bpp_prompt = '''When faced with a task, begin by identifying the brain regions that will contribute to solving the task. Depending on the complexity of the task, decide whether to assign broad categories of brain regions or divide them into more detailed subcategories. Then, initiate a multi-round collaboration process until a final solution is reached. The brain regions will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
--- 
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1

Participants: AI Assistant (you); Superior Parietal Lobule; Dorsolateral Prefrontal Cortex; Primary Visual Cortex; Hippocampus; Amygdala

Start collaboration! 

Superior Parietal Lobule: "Let’s analyze the task in detail. We need to use exactly the four numbers (6, 12, 1, 1) to construct 24. One approach is to consider common factors or combinations that can reach 24. For example, numbers like 12 and 6 are related by multiplication, but we should also explore addition or subtraction to hit the target. Let’s try to construct potential solutions step by step."
AI Assistant (you): "Thanks for the guidance! Here’s my initial solution: (12 / (1 + 1)) * 6 = 24."
Dorsolateral Prefrontal Cortex: "Let’s break that down: (1 + 1) = 2, (12 / 2) = 6, and 6 * 6 = 36. This solution is incorrect as it results in 36, not 24. We need to consider different combinations and avoid division in this way."
Primary Visual Cortex: "Visually, I see the error. Multiplying 6 by itself gives us too high of a result. Let’s imagine breaking down 24 into smaller components first, like through addition. For instance, 12 + 12 equals 24."
AI Assistant (you): "Thanks for pointing out the mistake. Here is a revised solution: (6 + 1 + 1) * (12 / 4) = 24."
Hippocampus: "Wait a minute, this is close, but you’ve introduced a 4, which is not part of the original numbers. Let’s avoid using extra numbers and stick strictly to the provided set: 6, 12, 1, and 1."
AI Assistant (you): "You’re right, I overlooked that. Here is a revised solution considering only the provided numbers: 6 * (1 - 1) + 12 = 24."
Amygdala: "Hmm, I appreciate the effort, but you’re very close! The subtraction might be causing the issue. Let’s try changing it to an addition. Keep up the great work, we’re almost there!"
AI Assistant (you): "Got it! Here’s the corrected solution: 6 * (1 + 1) + 12 = 24."
Superior Parietal Lobule: "Let’s check it step by step: 1 + 1 = 2, 6 * 2 = 12, and 12 + 12 = 24. The solution works, and all the numbers were used correctly."
Amygdala: "Great job! This feels satisfying now that we’ve nailed it. Excellent teamwork!"

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24.

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid. 

Participants: AI Assistant (you); Frontal Lobe; Temporal Lobe; Limbic System; Occipital Lobe; Hippocampus

Start collaboration!

Frontal Lobe: "We need to organize the poem into seven lines, with the first letters spelling 'CHATGPT.' The language should be simple enough for a ten-year-old, and we need to include basic explanations of quantum computing."
Temporal Lobe: "Let’s recall the basics of quantum computing: it’s a type of computer that uses qubits, allowing multiple states at once, unlike classical bits. I’ll provide this key information for the poem."
Hippocampus: "I recall some basic metaphors used in the past to explain complex ideas to children. Let’s use some of those to simplify the explanation."
Occipital Lobe: "We should include vivid imagery to make the poem engaging. For example, comparing qubits to spinning tops or magic coins might help."
Limbic System: "Let’s ensure the poem is fun and interesting for a child. Using playful words and a light tone will keep them engaged."
AI Assistant (you): "Thanks for the input! Here’s my first attempt:
Curious machine of our time,
Harnessing the quantum realm’s odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await."

Frontal Lobe: "The structure looks correct. The first letters spell out 'CHATGPT,' and the poem stays focused on the topic."
Temporal Lobe: "Good job! The information about quantum computing is accurate, but I think we can simplify it a bit more for children."
Limbic System: "The tone feels light, but we could make it even more playful. Maybe add a bit more fun language."
AI Assistant (you): "Got it! I’ll simplify and adjust the tone. Let me revise it:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!"

Frontal Lobe: "This version also maintains the structure with 'CHATGPT.' The tone feels lighter and more playful."
Temporal Lobe: "I agree. The explanation is even simpler now, and the playful language makes it more engaging for a child."
Hippocampus: "I like how this reflects previous simple metaphors that work well for children. The playful nature helps."
Limbic System: "I love the playful tone! It's much more fun and engaging now."
Occipital Lobe: "The imagery is vivid and clear, especially with 'juggling' and 'secrets.' This should help kids visualize the concept better."

Finish collaboration!

Final answer:
Curious machine, what a wonder!
Harnessing strange quantum thunder,
Atoms play, two states they juggle,
Taking secrets with a giggle.

Great at solving, fast and neat,
Puzzles that no one else can beat,
Tomorrow's future, what a treat!

---
Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Determine whether the given sentence contains the answer to the following question. Question: {question}, Sentence: {sentence}, Answer:
'''
