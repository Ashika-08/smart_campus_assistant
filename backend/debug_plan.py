from study_planner import generate_study_plan
import json

sample_text = """
Introduction to Machine Learning.
Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.
Supervised learning: The computer is presented with example inputs and their desired outputs, given by a "teacher", and the goal is to learn a general rule that maps inputs to outputs.
Unsupervised learning: No labels are given to the learning algorithm, leaving it on its own to find structure in its input.
Reinforcement learning: A computer program interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle or playing a game against an opponent).
Deep Learning: A subset of machine learning based on artificial neural networks with representation learning.
"""

print("--- Testing Study Planner Generation ---")
try:
    plan = generate_study_plan(sample_text, days=3)
    print("Successfully generated plan:")
    print(json.dumps(plan, indent=2))
except Exception as e:
    print(f"FAILED: {e}")
