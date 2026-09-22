import math
import numpy as np
from typing import List, Dict, Any

def calculate_angle(pt1, pt2):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    return math.degrees(math.atan2(dy, dx))

ALL_TRAITS = [
    {
        "name": "Communication",
        "strength_title": "Clear Expression",
        "overall": "You tend to communicate with clarity and intention, especially when you have time to structure your thoughts.",
        "best": "Clear, structured and thoughtful communication that inspires trust.",
        "watch": "Over-structuring can sometimes slow spontaneous conversations.",
        "action": "Ask one open-ended question before offering your solution."
    },
    {
        "name": "Decision Style",
        "strength_title": "Strategic Clarity",
        "overall": "You tend to evaluate options with a long-term perspective.",
        "best": "Strategic, visionary, and forward-looking.",
        "watch": "Can overthink simple decisions.",
        "action": "Trust your initial gut feeling on smaller choices."
    },
    {
        "name": "Adaptability",
        "strength_title": "Adaptive Agility",
        "overall": "You adjust quickly to new information and changing environments.",
        "best": "Flexible, resilient, and open-minded.",
        "watch": "Can lose focus on long-term goals.",
        "action": "Set one rigid milestone per week to anchor your flexibility."
    },
    {
        "name": "Execution",
        "strength_title": "Structured Execution",
        "overall": "You drive tasks to completion with intense focus.",
        "best": "Reliable, consistent, and highly organized.",
        "watch": "May struggle when plans change abruptly.",
        "action": "Build buffer time into your daily schedule."
    },
    {
        "name": "Creativity",
        "strength_title": "Innovative Thinking",
        "overall": "You approach problems from unique angles and value originality.",
        "best": "Imaginative, resourceful, and capable of connecting dots others miss.",
        "watch": "Can get lost in ideas without anchoring them to reality.",
        "action": "Jot down the practical steps required to test your best idea today."
    },
    {
        "name": "Empathy",
        "strength_title": "Deep Intuition",
        "overall": "You naturally sense the emotional undertones of a situation.",
        "best": "Compassionate, supportive, and a great listener.",
        "watch": "Can absorb others' stress and become overwhelmed.",
        "action": "Take 5 minutes of quiet time after a heavy conversation."
    },
    {
        "name": "Logic",
        "strength_title": "Analytical Rigor",
        "overall": "You break complex issues into logical, manageable pieces.",
        "best": "Objective, rational, and excellent at troubleshooting.",
        "watch": "May overlook the emotional impact of a purely logical decision.",
        "action": "Before presenting a solution, consider how it makes others feel."
    },
    {
        "name": "Focus",
        "strength_title": "Laser Concentration",
        "overall": "You have the ability to dive deep into a single task for extended periods.",
        "best": "Productive, detail-oriented, and resilient against distractions.",
        "watch": "Can develop tunnel vision and miss the broader context.",
        "action": "Set an alarm to step back and look at the big picture every hour."
    }
]

def generate_profile(boxes: List[Any], text: str) -> dict:
    if not boxes or len(boxes) == 0:
        return {
            "standout_strength": "Balanced Execution",
            "scores": [{"name": "Execution", "value": 75}],
            "insights": []
        }

    angles, heights, x_starts = [], [], []
    for item in boxes:
        pts = item[0]
        angles.append(calculate_angle(pts[0], pts[1]))
        heights.append(abs(pts[3][1] - pts[0][1]))
        x_starts.append(pts[0][0])

    avg_angle = np.mean(angles) if angles else 0
    height_var = np.var(heights) if len(heights) > 1 else 0
    margin_var = np.var(x_starts) if len(x_starts) > 1 else 0

    text_hash = hash(text) if text else 0
    
    # Pick 5 unique traits based on the hash
    selected_indices = []
    pool_copy = list(range(len(ALL_TRAITS)))
    for i in range(5):
        idx = (text_hash + i*13) % len(pool_copy)
        selected_indices.append(pool_copy.pop(idx))
        
    selected_traits = [ALL_TRAITS[i] for i in selected_indices]
    
    scores = []
    insights = []
    
    # Calculate geometric-based dynamic scores for the selected traits
    for i, trait in enumerate(selected_traits):
        jitter = ((text_hash >> (i * 2)) % 20) - 5
        
        # Base physical metrics influence different traits
        if trait["name"] in ["Decision Style", "Logic"]:
            base = 80 + (abs(avg_angle) % 15)
        elif trait["name"] in ["Adaptability", "Creativity"]:
            base = 80 + (height_var % 15)
        elif trait["name"] in ["Communication", "Empathy"]:
            base = 80 + (margin_var % 15)
        else:
            base = 80 + (len(text) % 15)
            
        score_val = int(min(max(base + jitter, 78), 98))
        
        scores.append({
            "name": trait["name"],
            "value": score_val,
            "strength_title": trait["strength_title"]
        })
        
        insights.append({
            "title": trait["name"],
            "overall_interpretation": trait["overall"],
            "at_your_best": trait["best"],
            "what_to_watch": trait["watch"],
            "action_to_practise": trait["action"]
        })

    # Standout strength is the one with the highest score
    standout_trait = max(scores, key=lambda x: x["value"])
    
    # Clean up scores to match Schema (remove strength_title)
    final_scores = [{"name": s["name"], "value": s["value"]} for s in scores]

    return {
        "standout_strength": standout_trait["strength_title"],
        "scores": final_scores,
        "insights": insights
    }
