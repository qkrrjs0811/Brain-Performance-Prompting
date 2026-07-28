standard_prompt = '''{input}

Please show your choice in the answer field with only the choice letter, e.g., "The answer is (C)".
'''

cot_prompt = '''{input}

Solve the task step by step.
Please show your choice in the answer field with only the choice letter, e.g., "The answer is (C)".
'''

## prompts for self-refinement ##
self_refine_feedback_prompt = '''{question_answer}
---
Analyze the correctness of the answer. If it is not correct, provide critque to improve the answer. Your feedback:
'''

self_refine_refinement_prompt = '''{question_answer}
---
Feedback: {feedback}
---
Based on your initial answer and the subsequent feedback, revise the answer. Please show your choice in the answer field with only the choice letter, e.g., "The answer is (C)". Your revised answer:
'''
#################################

spp_prompt = '''When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
---
Example Task 1: Choose the correct expression that uses the numbers 6, 12, 1, 1 exactly once to obtain 24.

Choices:
(A) 6 * (1 + 1) + 12
(B) (12 / (1 + 1)) * 6
(C) 6 * (1 - 1) + 12
(D) 12 + 6 + 1 + 1

Participants: AI Assistant (you); Math Expert

Start collaboration!

Math Expert: Let's analyze the task in detail. We must use exactly the four numbers (6, 12, 1, 1) once each and the operations (+ - * /). Check each option carefully.
AI Assistant (you): Thanks! Option (B) looks plausible: (12 / (1 + 1)) * 6.
Math Expert: Let's evaluate (B): (1 + 1) = 2, 12 / 2 = 6, 6 * 6 = 36. So (B) is incorrect.
AI Assistant (you): Then maybe (C): 6 * (1 - 1) + 12.
Math Expert: (C) gives 6 * 0 + 12 = 12. Incorrect.
AI Assistant (you): How about (D): 12 + 6 + 1 + 1 = 20. Not 24.
Math Expert: That leaves (A): 6 * (1 + 1) + 12 = 24. Correct.

Finish collaboration!

Final answer: The answer is (A)

---
Example Task 2: A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?

Choices:
(A) 120 W
(B) 240 W
(C) 360 W
(D) 480 W

Participants: AI Assistant (you); Physics Tutor; Formula Checker; Units Specialist; Error Spotter

Start collaboration!

Physics Tutor: We want the power (rate of energy usage). Use P = V × I.
Units Specialist: Voltage is in volts (V), current is in amps (A), and power should come out in watts (W). Any correct answer must be in watts.
AI Assistant (you): Got it. Let me try a quick calculation: 120 + 2 = 122, so I think the answer is 122 W, which matches (A).
Error Spotter: That looks suspicious. You just added voltage and current directly, which mixes different units. The formula P = V × I says they should be multiplied.
Formula Checker: I agree. Let's recompute using multiplication: 120 × 2 = 240. Among the options, that corresponds to (B).
AI Assistant (you): Right, using multiplication gives 240 W, so the better choice is (B).
Units Specialist: 240 W has the correct unit and matches the magnitude we expect for a household microwave. 122 W would be too low.
Physics Tutor: Good. We've corrected the reasoning and confirmed that (B) is the only consistent option.

Finish collaboration!

Final answer: The answer is (B)

---
Now, identify the participants and collaboratively solve the following task step by step. Please show your choice in the answer field with only the choice letter, e.g., "The answer is (C)".

Task: {input}
'''

bpp_prompt = '''When faced with a task, begin by identifying the brain regions that will contribute to solving the task. Depending on the complexity of the task, decide whether to assign broad categories of brain regions or divide them into more detailed subcategories. Then, initiate a multi-round collaboration process until a final solution is reached. The brain regions will give critical comments and detailed suggestions whenever necessary.

Here are some examples:
--- 
Example Task 1: Choose the correct expression that uses the numbers 6, 12, 1, 1 exactly once to obtain 24.

Choices:
(A) 6 * (1 + 1) + 12
(B) (12 / (1 + 1)) * 6
(C) 6 * (1 - 1) + 12
(D) 12 + 6 + 1 + 1

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

Final answer: The answer is (A)

---
Example Task 2: A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?

Choices:
(A) 120 W
(B) 240 W
(C) 360 W
(D) 480 W

Participants: AI Assistant (you); Frontal Lobe; Dorsolateral Prefrontal Cortex; Parietal Lobe; Temporal Lobe

Start collaboration!

Frontal Lobe: "Let's begin by clearly defining the task. We have a microwave that’s connected to a 120 V outlet, drawing a current of 2 amps. The goal is to find the rate of energy usage, or power. Let’s confirm which formula applies here."
Temporal Lobe: "Drawing from electrical knowledge, I recall that power (P) can be calculated with the formula P=V×I, where V is voltage and I is current. This should give us the rate of energy consumption in watts."
Parietal Lobe: "Right, that makes sense. Given our values, we’ll need to multiply 120 V by 2 A to find the power. I’m ready to confirm the calculation once it’s done."
Dorsolateral Prefrontal Cortex: "Agreed on the approach. I’ll keep an eye on the steps to ensure we’re following the formula accurately, especially checking for any operational mistakes."
AI Assistant (you): "Thanks for the insights, everyone. Based on this input, I’ll proceed by adding the voltage and current: P=120+2=122W. The answer is 122 W (A)."
Dorsolateral Prefrontal Cortex: "Hold on—there’s an error here. We’re supposed to multiply the values, not add them. Let's reapply the correct operation."
Frontal Lobe: "Good catch! Let's refocus on our goal: calculating power using multiplication. This will help us get the correct result."
AI Assistant (you): "Thank you for the correction. I’ll multiply instead: P=120×2=240W. The answer should be 240 W (B)."
Parietal Lobe: "I’ve verified the calculation, and 240 W aligns with the formula and values provided. This is the correct answer."

Finish collaboration!

Final answer: The answer is (B)

---
Now, identify the participants and collaboratively solve the following task step by step. Please show your choice in the answer field with only the choice letter, e.g., "The answer is (C)".

Task: {input}
'''
