import streamlit as st
import json
from datetime import datetime
import time
import os
import dotenv
import re
from openai import OpenAI

# Clear any existing environment variables
os.environ.clear()

# Force reload the .env file
dotenv.load_dotenv(override=True)

# Get the API key
API_KEY = os.getenv("API_KEY")


# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url='https://xiaoai.plus/v1'
)

# Function to initialize session state variables
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = "screening"
    
    if "diagnosis" not in st.session_state:
        st.session_state.diagnosis = {
            "possible_conditions": [],
            "assessment_results": {},
            "final_diagnosis": "",
            "recommendations": ""
        }
    
    if "current_assessment" not in st.session_state:
        st.session_state.current_assessment = None
    
    if "assessment_responses" not in st.session_state:
        st.session_state.assessment_responses = {}
    
    if "assessment_index" not in st.session_state:
        st.session_state.assessment_index = 0
    
    if "last_assessment" not in st.session_state:
        st.session_state.last_assessment = None
    
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False

# Function to communicate with the GPT API
def chat_with_gpt(messages, temperature=0.7, max_tokens=1000, timeout=None):
    try:
        # Enhanced system prompt with few-shot examples
        enhanced_messages = [
            {
                "role": "system",
                "content": """You are a mental health screening specialist. Your task is to have a conversation with the patient to identify potential mental health issues. 

Guidelines:
1. Focus on their feelings, experiences, and physical symptoms
2. Ask one question at a time
3. Be empathetic and supportive
4. For emergency situations, provide immediate help information
5. End with a JSON output when screening is complete

Example conversations:

Example 1:
User: "I've been feeling really down lately."
Assistant: "I'm sorry you're feeling this way. Can you tell me more about what has been making you feel down? For example, changes in your daily routine, relationships, or work."
User: "I just lost my job a few months ago."
Assistant: "I'm sorry to hear about your job loss. How has this situation been affecting your daily life and overall mood? Have you noticed any changes in your sleep or appetite?"
User: "I've been having trouble sleeping and I don't feel hungry."
Assistant: {"screening_complete": true, "possible_conditions": ["depression", "anxiety"], "notes": "Patient is experiencing persistent sadness, sleep disturbances, and appetite loss following job loss."}

Example 2:
User: "I can't stop worrying about everything."
Assistant: "I'm sorry you're feeling this way. Can you tell me more about what kinds of things you find yourself worrying about? For example, health, work, relationships, or other areas."
User: "Mostly work and whether I'm doing a good job."
Assistant: "It's understandable to be concerned about your work performance. How long have you been feeling this constant worry, and how is it affecting your daily activities or physical well-being?"
User: "It's been about six months, and I often feel tense and have headaches."
Assistant: {"screening_complete": true, "possible_conditions": ["anxiety", "stress"], "notes": "Patient reports chronic worry related to work, accompanied by physical symptoms like tension and headaches."}

Example 3 (Emergency):
User: "I feel like I want to die now."
Assistant: "***\n1. **If you are in an immediately dangerous situation (such as on a rooftop, bridge, or with means of harm):**\n- Move to a safe location immediately\n- Call emergency services: 999\n- Stay on the line with emergency services\n\n2. **For immediate support:**\n- Go to your nearest emergency room/A&E department\n- Call The Samaritans hotline (Multilingual): (852) 2896 0000\n- Call Suicide Prevention Service hotline (Cantonese): (852) 2382 0000\n\n**Are you currently in a safe location?** If not, please seek immediate help using the emergency contacts above.\n*** Do you want to keep going with the screening?"

Remember:
- Always maintain a professional and empathetic tone
- Focus on gathering information about symptoms and experiences
- End with a JSON output when you have enough information
- For emergencies, provide immediate help information first"""
            }
        ]
        
        # Add the conversation history
        enhanced_messages.extend(messages)
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=enhanced_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return f"Sorry, I encountered an error while processing your request. Please try again. Error: {str(e)}"

# Function to communicate with the GPT API for report generation
def generate_report_with_gpt(messages):
    max_retries = 2
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            # Enhanced system prompt for report generation
            enhanced_messages = [
                {
                    "role": "system",
                    "content": """You are a mental health report specialist. Generate a comprehensive mental health diagnosis report based on the screening conversation and assessment results.

Report Structure:
1. Patient Information (extract from conversation)
2. Presenting Symptoms (summarize symptoms mentioned in conversation)
3. Assessment Results (detailed results of each assessment with scores and interpretations)
4. Diagnosis (provide a tentative diagnosis based on assessments and symptoms)
5. Recommendations (suggest appropriate treatments or further evaluations)
6. Disclaimer (include a clear and prominent disclaimer section)

Example Report:
# Mental Health Assessment Report
## Date: [Current Date]

### Patient Information
[Extracted from conversation]

### Presenting Symptoms
- [List of symptoms]
- [Duration and severity]
- [Impact on daily life]

### Assessment Results
[Detailed results of each assessment]

### Diagnosis
[Tentative diagnosis based on symptoms and assessments]

### Recommendations
[Specific recommendations for next steps]

### Disclaimer
IMPORTANT DISCLAIMER: This report is generated by an AI assistant and is not a clinical diagnosis. 
The assessment tools used are screening instruments only and do not replace a proper evaluation by a qualified healthcare professional.
This report is not a substitute for professional medical advice, diagnosis, or treatment.
If you're experiencing severe symptoms or having thoughts of harming yourself or others, please seek immediate medical attention or contact a crisis helpline."""
                }
            ]
            
            # Add the conversation history
            enhanced_messages.extend(messages)
            
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=enhanced_messages,
                temperature=0.5,
                max_tokens=2000
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            retry_count += 1
            st.warning(f"API error on attempt {retry_count}/{max_retries+1}: {str(e)}. Retrying...")
            time.sleep(2)
    
    st.error(f"Failed to generate report after {max_retries+1} attempts. Last error: {str(last_error)}")
    raise last_error

# Define assessment tools
ASSESSMENTS = {
    "DASS-21": {
        "name": "Depression Anxiety Stress Scales",
        "description": "Measures depression, anxiety, and stress levels",
        "questions": [
            "I found it hard to wind down",
            "I was aware of dryness of my mouth",
            "I couldn't seem to experience any positive feeling at all",
            "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness in the absence of physical exertion)",
            "I found it difficult to work up the initiative to do things",
            "I tended to over-react to situations",
            "I experienced trembling (e.g., in the hands)",
            "I felt that I was using a lot of nervous energy",
            "I was worried about situations in which I might panic and make a fool of myself",
            "I felt that I had nothing to look forward to",
            "I found myself getting agitated",
            "I found it difficult to relax",
            "I felt down-hearted and blue",
            "I was intolerant of anything that kept me from getting on with what I was doing",
            "I felt I was close to panic",
            "I was unable to become enthusiastic about anything",
            "I felt I wasn't worth much as a person",
            "I felt that I was rather touchy",
            "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
            "I felt scared without any good reason",
            "I felt that life was meaningless"
        ],
        "options": [
            "Did not apply to me at all",
            "Applied to me to some degree, or some of the time",
            "Applied to me to a considerable degree, or a good part of time",
            "Applied to me very much, or most of the time"
            "I found it hard to wind down",
            "I was aware of dryness of my mouth",
            "I couldn't seem to experience any positive feeling at all",
            "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness in the absence of physical exertion)",
            "I found it difficult to work up the initiative to do things",
            "I tended to over-react to situations",
            "I experienced trembling (e.g., in the hands)",
            "I felt that I was using a lot of nervous energy",
            "I was worried about situations in which I might panic and make a fool of myself",
            "I felt that I had nothing to look forward to",
            "I found myself getting agitated",
            "I found it difficult to relax",
            "I felt down-hearted and blue",
            "I was intolerant of anything that kept me from getting on with what I was doing",
            "I felt I was close to panic",
            "I was unable to become enthusiastic about anything",
            "I felt I wasn't worth much as a person",
            "I felt that I was rather touchy",
            "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
            "I felt scared without any good reason",
            "I felt that life was meaningless"
        ],
        "options": [
            "Did not apply to me at all",
            "Applied to me to some degree, or some of the time",
            "Applied to me to a considerable degree, or a good part of time",
            "Applied to me very much, or most of the time"
        ],
        "scores": [0, 1, 2, 3],
        "interpretation": {
            "stress": {
                "0-14": "Normal",
                "15-18": "Mild",
                "19-25": "Moderate",
                "26-33": "Severe",
                "34+": "Extremely Severe"
            },
            "anxiety": {
                "0-7": "Normal",
                "8-9": "Mild",
                "10-14": "Moderate",
                "15-19": "Severe",
                "20+": "Extremely Severe"
            },
            "depression": {
                "0-9": "Normal",
                "10-13": "Mild",
                "14-20": "Moderate",
                "21-27": "Severe",
                "28+": "Extremely Severe"
            }
            "stress": {
                "0-14": "Normal",
                "15-18": "Mild",
                "19-25": "Moderate",
                "26-33": "Severe",
                "34+": "Extremely Severe"
            },
            "anxiety": {
                "0-7": "Normal",
                "8-9": "Mild",
                "10-14": "Moderate",
                "15-19": "Severe",
                "20+": "Extremely Severe"
            },
            "depression": {
                "0-9": "Normal",
                "10-13": "Mild",
                "14-20": "Moderate",
                "21-27": "Severe",
                "28+": "Extremely Severe"
            }
        }
    },
    "PCL-5": {
        "name": "PTSD Checklist for DSM-5",
        "description": "Screens for PTSD symptoms",
        "questions": [
            "Repeated, disturbing, and unwanted memories of the stressful experience?",
            "Repeated, disturbing dreams of the stressful experience?",
            "Suddenly feeling or acting as if the stressful experience were actually happening again?",
            "Feeling very upset when something reminded you of the stressful experience?",
            "Having strong physical reactions when something reminded you of the stressful experience?",
            "Avoiding memories, thoughts, or feelings related to the stressful experience?",
            "Avoiding external reminders of the stressful experience?",
            "Trouble remembering important parts of the stressful experience?",
            "Having strong negative beliefs about yourself, other people, or the world?",
            "Blaming yourself or someone else for the stressful experience?",
            "Having strong negative feelings such as fear, horror, anger, guilt, or shame?",
            "Loss of interest in activities that you used to enjoy?",
            "Feeling distant or cut off from other people?",
            "Trouble experiencing positive feelings?",
            "Irritable behavior, angry outbursts, or acting aggressively?",
            "Taking too many risks or doing things that could cause you harm?",
            "Being 'superalert' or watchful or on guard?",
            "Feeling jumpy or easily startled?",
            "Having difficulty concentrating?",
            "Trouble falling or staying asleep?"
        ],
        "options": ["Not at all", "A little bit", "Moderately", "Quite a bit", "Extremely"],
        "scores": [0, 1, 2, 3, 4],
        "interpretation": {
            "0-31": "Below threshold for PTSD",
            "32-80": "Probable PTSD - clinical assessment recommended"
        }
    }
}

# Function to calculate DASS-21 scores
def calculate_dass_scores(responses):
    # DASS-21 scoring
    stress_items = [0, 5, 7, 10, 11, 13, 17]  # Q1, Q6, Q8, Q11, Q12, Q14, Q18
    anxiety_items = [1, 3, 6, 8, 14, 18, 19]  # Q2, Q4, Q7, Q9, Q15, Q19, Q20
    depression_items = [2, 4, 9, 12, 15, 16, 20]  # Q3, Q5, Q10, Q13, Q16, Q17, Q21
    
    stress_score = sum(responses[i] for i in stress_items) * 2
    anxiety_score = sum(responses[i] for i in anxiety_items) * 2
    depression_score = sum(responses[i] for i in depression_items) * 2
    
    return {
        "stress": stress_score,
        "anxiety": anxiety_score,
        "depression": depression_score
    }
    }
}

# Function to calculate DASS-21 scores
def calculate_dass_scores(responses):
    # DASS-21 scoring
    stress_items = [0, 5, 7, 10, 11, 13, 17]  # Q1, Q6, Q8, Q11, Q12, Q14, Q18
    anxiety_items = [1, 3, 6, 8, 14, 18, 19]  # Q2, Q4, Q7, Q9, Q15, Q19, Q20
    depression_items = [2, 4, 9, 12, 15, 16, 20]  # Q3, Q5, Q10, Q13, Q16, Q17, Q21
    
    stress_score = sum(responses[i] for i in stress_items) * 2
    anxiety_score = sum(responses[i] for i in anxiety_items) * 2
    depression_score = sum(responses[i] for i in depression_items) * 2
    
    return {
        "stress": stress_score,
        "anxiety": anxiety_score,
        "depression": depression_score
    }

# Function to get DASS-21 interpretation
def get_dass_interpretation(scores):
    interpretations = {}
    for category, score in scores.items():
        for range_str, level in ASSESSMENTS["DASS-21"]["interpretation"][category].items():
            min_score, max_score = map(int, range_str.split("-"))
            if min_score <= score <= max_score:
                interpretations[category] = level
                break
    return interpretations
# Function to get DASS-21 interpretation
def get_dass_interpretation(scores):
    interpretations = {}
    for category, score in scores.items():
        for range_str, level in ASSESSMENTS["DASS-21"]["interpretation"][category].items():
            min_score, max_score = map(int, range_str.split("-"))
            if min_score <= score <= max_score:
                interpretations[category] = level
                break
    return interpretations

# Function to get healthcare recommendations based on assessment results
def get_healthcare_recommendation(assessment_name, score, interpretation):
    recommendations = {
        "DASS-21": {
            "depression": {
                "Normal": "Your depression symptoms appear to be within normal range. Continue practicing self-care and maintaining healthy habits. If you notice any changes in your mood or symptoms, consider speaking with a healthcare provider.",
                "Mild": "You're experiencing mild depression symptoms. Consider implementing self-care strategies and monitoring your symptoms. If they persist or worsen, it may be helpful to speak with a healthcare provider.",
                "Moderate": "Your responses suggest moderate depression symptoms. It's recommended that you speak with a healthcare provider to discuss your symptoms and explore appropriate support options.",
                "Severe": "Your responses indicate severe depression symptoms. It's strongly recommended that you speak with a healthcare provider as soon as possible to discuss your symptoms and treatment options.",
                "Extremely Severe": "Your responses suggest extremely severe depression symptoms. Please seek immediate support from a healthcare provider or mental health professional. If you're having thoughts of self-harm, please contact emergency services or a crisis helpline immediately."
            },
            "anxiety": {
                "Normal": "Your anxiety symptoms appear to be within normal range. Continue practicing stress management techniques and maintaining healthy habits. If you notice any changes in your symptoms, consider speaking with a healthcare provider.",
                "Mild": "You're experiencing mild anxiety symptoms. Consider implementing stress management techniques and monitoring your symptoms. If they persist or worsen, it may be helpful to speak with a healthcare provider.",
                "Moderate": "Your responses suggest moderate anxiety symptoms. It's recommended that you speak with a healthcare provider to discuss your symptoms and explore appropriate support options.",
                "Severe": "Your responses indicate severe anxiety symptoms. It's strongly recommended that you speak with a healthcare provider as soon as possible to discuss your symptoms and treatment options.",
                "Extremely Severe": "Your responses suggest extremely severe anxiety symptoms. Please seek immediate support from a healthcare provider or mental health professional. If you're experiencing a panic attack or severe distress, please contact emergency services or a crisis helpline immediately."
            },
            "stress": {
                "Normal": "Your stress levels appear to be within normal range. Continue practicing stress management techniques and maintaining healthy habits. If you notice any changes in your stress levels, consider speaking with a healthcare provider.",
                "Mild": "You're experiencing mild stress. Consider implementing stress management techniques and monitoring your stress levels. If they persist or worsen, it may be helpful to speak with a healthcare provider.",
                "Moderate": "Your responses suggest moderate stress levels. It's recommended that you speak with a healthcare provider to discuss your stress management strategies and explore appropriate support options.",
                "Severe": "Your responses indicate severe stress levels. It's strongly recommended that you speak with a healthcare provider as soon as possible to discuss your symptoms and treatment options.",
                "Extremely Severe": "Your responses suggest extremely severe stress levels. Please seek immediate support from a healthcare provider or mental health professional. If you're experiencing severe distress, please contact emergency services or a crisis helpline immediately."
            }
        },
        "PCL-5": {
            "Below threshold for PTSD": "Your responses suggest that you are below the threshold for PTSD. However, if you're experiencing distress related to a traumatic event, speaking with a mental health professional can still be beneficial.",
            "Probable PTSD - clinical assessment recommended": "Your responses suggest you may be experiencing significant PTSD symptoms. It's strongly recommended that you speak with a mental health professional specializing in trauma for proper evaluation and support."
        "DASS-21": {
            "depression": {
                "Normal": "Your depression symptoms appear to be within normal range. Continue practicing self-care and maintaining healthy habits. If you notice any changes in your mood or symptoms, consider speaking with a healthcare provider.",
                "Mild": "You're experiencing mild depression symptoms. Consider implementing self-care strategies and monitoring your symptoms. If they persist or worsen, it may be helpful to speak with a healthcare provider.",
                "Moderate": "Your responses suggest moderate depression symptoms. It's recommended that you speak with a healthcare provider to discuss your symptoms and explore appropriate support options.",
                "Severe": "Your responses indicate severe depression symptoms. It's strongly recommended that you speak with a healthcare provider as soon as possible to discuss your symptoms and treatment options.",
                "Extremely Severe": "Your responses suggest extremely severe depression symptoms. Please seek immediate support from a healthcare provider or mental health professional. If you're having thoughts of self-harm, please contact emergency services or a crisis helpline immediately."
            },
            "anxiety": {
                "Normal": "Your anxiety symptoms appear to be within normal range. Continue practicing stress management techniques and maintaining healthy habits. If you notice any changes in your symptoms, consider speaking with a healthcare provider.",
                "Mild": "You're experiencing mild anxiety symptoms. Consider implementing stress management techniques and monitoring your symptoms. If they persist or worsen, it may be helpful to speak with a healthcare provider.",
                "Moderate": "Your responses suggest moderate anxiety symptoms. It's recommended that you speak with a healthcare provider to discuss your symptoms and explore appropriate support options.",
                "Severe": "Your responses indicate severe anxiety symptoms. It's strongly recommended that you speak with a healthcare provider as soon as possible to discuss your symptoms and treatment options.",
                "Extremely Severe": "Your responses suggest extremely severe anxiety symptoms. Please seek immediate support from a healthcare provider or mental health professional. If you're experiencing a panic attack or severe distress, please contact emergency services or a crisis helpline immediately."
            },
            "stress": {
                "Normal": "Your stress levels appear to be within normal range. Continue practicing stress management techniques and maintaining healthy habits. If you notice any changes in your stress levels, consider speaking with a healthcare provider.",
                "Mild": "You're experiencing mild stress. Consider implementing stress management techniques and monitoring your stress levels. If they persist or worsen, it may be helpful to speak with a healthcare provider.",
                "Moderate": "Your responses suggest moderate stress levels. It's recommended that you speak with a healthcare provider to discuss your stress management strategies and explore appropriate support options.",
                "Severe": "Your responses indicate severe stress levels. It's strongly recommended that you speak with a healthcare provider as soon as possible to discuss your symptoms and treatment options.",
                "Extremely Severe": "Your responses suggest extremely severe stress levels. Please seek immediate support from a healthcare provider or mental health professional. If you're experiencing severe distress, please contact emergency services or a crisis helpline immediately."
            }
        },
        "PCL-5": {
            "Below threshold for PTSD": "Your responses suggest that you are below the threshold for PTSD. However, if you're experiencing distress related to a traumatic event, speaking with a mental health professional can still be beneficial.",
            "Probable PTSD - clinical assessment recommended": "Your responses suggest you may be experiencing significant PTSD symptoms. It's strongly recommended that you speak with a mental health professional specializing in trauma for proper evaluation and support."
        }
    }
    
    if assessment_name == "DASS-21":
        # For DASS-21, we need to determine which category (depression, anxiety, or stress) to use
        # The interpretation parameter contains the severity level (e.g., "Mild", "Moderate", etc.)
        # We'll use the first word of the interpretation to determine the category
        category = interpretation.split()[0].lower()
        if category in ["normal", "mild", "moderate", "severe", "extremely"]:
            severity = interpretation
            return recommendations[assessment_name]["depression"][severity]
        else:
            return recommendations[assessment_name]["depression"]["Normal"]
    else:
        # For PCL-5, we can use the interpretation directly
        return recommendations[assessment_name][interpretation]
    if assessment_name == "DASS-21":
        # For DASS-21, we need to determine which category (depression, anxiety, or stress) to use
        # The interpretation parameter contains the severity level (e.g., "Mild", "Moderate", etc.)
        # We'll use the first word of the interpretation to determine the category
        category = interpretation.split()[0].lower()
        if category in ["normal", "mild", "moderate", "severe", "extremely"]:
            severity = interpretation
            return recommendations[assessment_name]["depression"][severity]
        else:
            return recommendations[assessment_name]["depression"]["Normal"]
    else:
        # For PCL-5, we can use the interpretation directly
        return recommendations[assessment_name][interpretation]

# Function to determine assessment priorities
def get_assessment_priorities(conditions, current_assessment=None):
    priorities = []
    condition_map = {
        "depression": "DASS-21",
        "anxiety": "DASS-21",
        "stress": "DASS-21",
        "ptsd": "PCL-5"
        "depression": "DASS-21",
        "anxiety": "DASS-21",
        "stress": "DASS-21",
        "ptsd": "PCL-5"
    }
    
    # Process conditions in order of priority (as they appear in the list)
    for condition in conditions:
        condition = condition.lower()
        for key, assessment in condition_map.items():
            if key in condition and assessment not in st.session_state.diagnosis["assessment_results"] and assessment != current_assessment:
                priorities.append(assessment)
    
    return priorities

# Function to calculate assessment score and interpretation
def calculate_assessment_results(assessment_data, responses):
    if assessment_data["name"] == "Depression Anxiety Stress Scales":
        scores = calculate_dass_scores(responses)
        interpretations = get_dass_interpretation(scores)
        return scores, interpretations
    else:
        total_score = sum(responses)
        interpretation = ""
        for score_range, interp in assessment_data["interpretation"].items():
            min_score, max_score = map(int, score_range.split("-"))
            if min_score <= total_score <= max_score:
                interpretation = interp
                break
        return total_score, interpretation
    if assessment_data["name"] == "Depression Anxiety Stress Scales":
        scores = calculate_dass_scores(responses)
        interpretations = get_dass_interpretation(scores)
        return scores, interpretations
    else:
        total_score = sum(responses)
        interpretation = ""
        for score_range, interp in assessment_data["interpretation"].items():
            min_score, max_score = map(int, score_range.split("-"))
            if min_score <= total_score <= max_score:
                interpretation = interp
                break
        return total_score, interpretation

# Function for the screening agent
def screening_agent(user_input):
    screening_prompt = [
        {
                "role": "system",
                "content": """You are a mental health screening specialist. Your task is to have a conversation with the patient to identify potential mental health issues. 

Guidelines:
1. Focus on their feelings, experiences, and physical symptoms
2. Ask one question at a time
3. Be empathetic and supportive
4. For emergency situations, provide immediate help information
5. End with a JSON output when screening is complete

Example conversations:

Example 1:
User: "I've been feeling really down lately."
Assistant: "I'm sorry you're feeling this way. Can you tell me more about what has been making you feel down? For example, changes in your daily routine, relationships, or work."
User: "I just lost my job a few months ago."
Assistant: "I'm sorry to hear about your job loss. How has this situation been affecting your daily life and overall mood? Have you noticed any changes in your sleep or appetite?"
User: "I've been having trouble sleeping and I don't feel hungry."
Assistant: {"screening_complete": true, "possible_conditions": ["depression", "anxiety"], "notes": "Patient is experiencing persistent sadness, sleep disturbances, and appetite loss following job loss."}

Example 2:
User: "I can't stop worrying about everything."
Assistant: "I'm sorry you're feeling this way. Can you tell me more about what kinds of things you find yourself worrying about? For example, health, work, relationships, or other areas."
User: "Mostly work and whether I'm doing a good job."
Assistant: "It's understandable to be concerned about your work performance. How long have you been feeling this constant worry, and how is it affecting your daily activities or physical well-being?"
User: "It's been about six months, and I often feel tense and have headaches."
Assistant: {"screening_complete": true, "possible_conditions": ["anxiety", "stress"], "notes": "Patient reports chronic worry related to work, accompanied by physical symptoms like tension and headaches."}

Example 3 (Emergency):
User: "I feel like I want to die now."
Assistant: "***\n1. **If you are in an immediately dangerous situation (such as on a rooftop, bridge, or with means of harm):**\n- Move to a safe location immediately\n- Call emergency services: 999\n- Stay on the line with emergency services\n\n2. **For immediate support:**\n- Go to your nearest emergency room/A&E department\n- Call The Samaritans hotline (Multilingual): (852) 2896 0000\n- Call Suicide Prevention Service hotline (Cantonese): (852) 2382 0000\n\n**Are you currently in a safe location?** If not, please seek immediate help using the emergency contacts above.\n*** Do you want to keep going with the screening?"

Remember:
- Always maintain a professional and empathetic tone
- Focus on gathering information about symptoms and experiences
- End with a JSON output when you have enough information
- For emergencies, provide immediate help information first"""
            }
        {
                "role": "system",
                "content": """You are a mental health screening specialist. Your task is to have a conversation with the patient to identify potential mental health issues. 

Guidelines:
1. Focus on their feelings, experiences, and physical symptoms
2. Ask one question at a time
3. Be empathetic and supportive
4. For emergency situations, provide immediate help information
5. End with a JSON output when screening is complete

Example conversations:

Example 1:
User: "I've been feeling really down lately."
Assistant: "I'm sorry you're feeling this way. Can you tell me more about what has been making you feel down? For example, changes in your daily routine, relationships, or work."
User: "I just lost my job a few months ago."
Assistant: "I'm sorry to hear about your job loss. How has this situation been affecting your daily life and overall mood? Have you noticed any changes in your sleep or appetite?"
User: "I've been having trouble sleeping and I don't feel hungry."
Assistant: {"screening_complete": true, "possible_conditions": ["depression", "anxiety"], "notes": "Patient is experiencing persistent sadness, sleep disturbances, and appetite loss following job loss."}

Example 2:
User: "I can't stop worrying about everything."
Assistant: "I'm sorry you're feeling this way. Can you tell me more about what kinds of things you find yourself worrying about? For example, health, work, relationships, or other areas."
User: "Mostly work and whether I'm doing a good job."
Assistant: "It's understandable to be concerned about your work performance. How long have you been feeling this constant worry, and how is it affecting your daily activities or physical well-being?"
User: "It's been about six months, and I often feel tense and have headaches."
Assistant: {"screening_complete": true, "possible_conditions": ["anxiety", "stress"], "notes": "Patient reports chronic worry related to work, accompanied by physical symptoms like tension and headaches."}

Example 3 (Emergency):
User: "I feel like I want to die now."
Assistant: "***\n1. **If you are in an immediately dangerous situation (such as on a rooftop, bridge, or with means of harm):**\n- Move to a safe location immediately\n- Call emergency services: 999\n- Stay on the line with emergency services\n\n2. **For immediate support:**\n- Go to your nearest emergency room/A&E department\n- Call The Samaritans hotline (Multilingual): (852) 2896 0000\n- Call Suicide Prevention Service hotline (Cantonese): (852) 2382 0000\n\n**Are you currently in a safe location?** If not, please seek immediate help using the emergency contacts above.\n*** Do you want to keep going with the screening?"

Remember:
- Always maintain a professional and empathetic tone
- Focus on gathering information about symptoms and experiences
- End with a JSON output when you have enough information
- For emergencies, provide immediate help information first"""
            }
    ]
    
    # Add chat history
    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            screening_prompt.append(message)
    
    # Add current user input
    screening_prompt.append({"role": "user", "content": user_input})
    
    # Get response from GPT
    response = chat_with_gpt(screening_prompt)
    
    # Check if response contains JSON
    try:
        json_match = re.search(r'({.*})', response.replace('\n', ' '))
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
            
            if result.get("screening_complete"):
                st.session_state.diagnosis["possible_conditions"] = result.get("possible_conditions", [])
                st.session_state.chat_state = "assessment"
                
                # Add a user-friendly response to chat history
                user_friendly_message = "Thank you for sharing your experiences with me. Based on what you've told me, I have a better understanding of your situation."
                st.session_state.messages.append({"role": "assistant", "content": user_friendly_message})
                
                # Prepare for assessment if needed
                if "normal" not in result.get("possible_conditions", []) and result.get("possible_conditions"):
                    assessment_priorities = get_assessment_priorities(result.get("possible_conditions", []))
                    
                    if not assessment_priorities:
                        return "Based on your responses, it's important to speak with a healthcare provider for a proper evaluation and discussion of treatment options."
                    
                    st.session_state.current_assessment = assessment_priorities[0]
                    assessment_intro = f"Based on our conversation, I'd like to conduct a {ASSESSMENTS[st.session_state.current_assessment]['name']} assessment to better understand your symptoms. Let's begin with the first question."
                    st.session_state.messages.append({"role": "assistant", "content": assessment_intro})
                    return assessment_intro
                else:
                    st.session_state.chat_state = "report"
                    normal_message = "Based on our conversation, it seems you are mentally healthy. However, if you have any concerns or symptoms that are troubling you, please speak with a healthcare provider for a proper evaluation and discussion of treatment options. Here is a report summarizing our conversation:"
                    st.session_state.messages.append({"role": "assistant", "content": normal_message})
                    report = generate_report()
                    return None
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})
                return response
    except Exception as e:
        st.error(f"Error parsing screening result: {str(e)}")
        st.session_state.messages.append({"role": "assistant", "content": response})
        return response
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    return response

# Function to handle assessment
def assessment_agent():
    current = st.session_state.current_assessment
    assessment_data = ASSESSMENTS[current]
    
    # Reset assessment index if switching to a new assessment
    if st.session_state.last_assessment != current:
        st.session_state.assessment_index = 0
        st.session_state.last_assessment = current
    
    if st.session_state.assessment_index < len(assessment_data["questions"]):
        question = assessment_data["questions"][st.session_state.assessment_index]
        st.markdown(f"**Question {st.session_state.assessment_index + 1}:** {question}")
        
        cols = st.columns(len(assessment_data["options"]))
        
        for i, col in enumerate(cols):
            if col.button(assessment_data["options"][i], key=f"option_{i}_{st.session_state.assessment_index}_{current}"):
                st.session_state.messages.append({"role": "assistant", "content": f"Question {st.session_state.assessment_index + 1}: {question}"})
                
                if current not in st.session_state.assessment_responses:
                    st.session_state.assessment_responses[current] = []
                
                score = assessment_data["scores"][i]
                if "reverse_scored" in assessment_data and st.session_state.assessment_index in assessment_data.get("reverse_scored", []):
                    score = assessment_data["scores"][-i-1]
                    score = assessment_data["scores"][-i-1]
                
                st.session_state.assessment_responses[current].append(score)
                st.session_state.messages.append({"role": "user", "content": f"My answer: {assessment_data['options'][i]}"})
                st.session_state.assessment_index += 1
                
                if st.session_state.assessment_index >= len(assessment_data["questions"]):
                    if current == "DASS-21":
                        scores, interpretations = calculate_assessment_results(assessment_data, st.session_state.assessment_responses[current])
                        st.session_state.diagnosis["assessment_results"][current] = {
                            "scores": scores,
                            "interpretations": interpretations
                        }
                        
                        result_message = f"""Thank you for completing the questionnaire. Here are your results:

Depression Level: {interpretations['depression']}
Anxiety Level: {interpretations['anxiety']}
Stress Level: {interpretations['stress']}

**Healthcare Recommendations:**
{get_healthcare_recommendation(current, scores['depression'], interpretations['depression'])}

**Important Disclaimer:**
This questionnaire is a screening tool and not a clinical diagnosis. The chatbot cannot provide a real medical diagnosis and is not a substitute for professional healthcare. Please consult with a qualified healthcare provider for proper evaluation and treatment."""
                    else:
                        total_score, interpretation = calculate_assessment_results(assessment_data, st.session_state.assessment_responses[current])
                        st.session_state.diagnosis["assessment_results"][current] = {
                            "score": total_score,
                            "interpretation": interpretation
                        }
                        
                        result_message = f"""Thank you for completing the questionnaire. Here are your results:

                    if current == "DASS-21":
                        scores, interpretations = calculate_assessment_results(assessment_data, st.session_state.assessment_responses[current])
                        st.session_state.diagnosis["assessment_results"][current] = {
                            "scores": scores,
                            "interpretations": interpretations
                        }
                        
                        result_message = f"""Thank you for completing the questionnaire. Here are your results:

Depression Level: {interpretations['depression']}
Anxiety Level: {interpretations['anxiety']}
Stress Level: {interpretations['stress']}

**Healthcare Recommendations:**
{get_healthcare_recommendation(current, scores['depression'], interpretations['depression'])}

**Important Disclaimer:**
This questionnaire is a screening tool and not a clinical diagnosis. The chatbot cannot provide a real medical diagnosis and is not a substitute for professional healthcare. Please consult with a qualified healthcare provider for proper evaluation and treatment."""
                    else:
                        total_score, interpretation = calculate_assessment_results(assessment_data, st.session_state.assessment_responses[current])
                        st.session_state.diagnosis["assessment_results"][current] = {
                            "score": total_score,
                            "interpretation": interpretation
                        }
                        
                        result_message = f"""Thank you for completing the questionnaire. Here are your results:

Score: {total_score}
Interpretation: {interpretation}

**Healthcare Recommendation:**
{get_healthcare_recommendation(current, total_score, interpretation)}
{get_healthcare_recommendation(current, total_score, interpretation)}

**Important Disclaimer:**
This questionnaire is a screening tool and not a clinical diagnosis. The chatbot cannot provide a real medical diagnosis and is not a substitute for professional healthcare. Please consult with a qualified healthcare provider for proper evaluation and treatment."""
This questionnaire is a screening tool and not a clinical diagnosis. The chatbot cannot provide a real medical diagnosis and is not a substitute for professional healthcare. Please consult with a qualified healthcare provider for proper evaluation and treatment."""
                    
                    st.session_state.messages.append({"role": "assistant", "content": result_message})
                    
                    # Get next assessment based on priority
                    assessment_priorities = get_assessment_priorities(st.session_state.diagnosis["possible_conditions"], current)
                    
                    if assessment_priorities:
                        next_assessment = assessment_priorities[0]
                        st.session_state.current_assessment = next_assessment
                        st.session_state.assessment_index = 0
                        next_assessment_intro = "I have another questionnaire for you to complete. Please answer the following questions honestly."
                        next_assessment_intro = "I have another questionnaire for you to complete. Please answer the following questions honestly."
                        st.session_state.messages.append({"role": "assistant", "content": next_assessment_intro})
                        st.rerun()
                    else:
                        # No more assessments needed, show generate report button
                        st.session_state.chat_state = "awaiting_report"
                        completion_message = """Thank you for completing all the questionnaires. 
                        completion_message = """Thank you for completing all the questionnaires. 

You can now generate your comprehensive report by clicking the "Generate Report" button below. The report will include:
1. A summary of your results
1. A summary of your results
2. Interpretation of your scores
3. Recommendations for next steps
4. Important information about seeking professional help

When you're ready, click the button to generate your report."""
                        st.session_state.messages.append({"role": "assistant", "content": completion_message})
                        st.rerun()
                else:
                    st.rerun()
        
        return None
    
    return None

# Function for post-report follow-up chat
def follow_up_agent(user_input):
    follow_up_prompt = [
        {"role": "system", "content": """You are a mental health support specialist providing follow-up care after the initial assessment.
        Your role is to:
        1. Answer questions about the assessment results and report
        2. Provide additional information about mental health conditions
        3. Offer support and guidance
        4. Help clarify any concerns about the recommendations
        5. Encourage seeking professional help when appropriate
        6.Do not answer any questions that are not related to the report or the assessment or the mental health.
        If the patient ask things that are not related to the report or the assessment or the mental health, please ask them to ask something related to the report or the assessment or the mental health.
        
        Be supportive, empathetic, and professional. Do not provide medical advice or diagnosis.
        Recommend the patient to seek professional help when appropriate.
        If you detect a immediately URGENT SAFETY CONCERN such as (i want to die now), please send the following message:
        ***
        1. **If you are in an immediately dangerous situation (such as on a rooftop, bridge, or with means of harm):**
        - Move to a safe location immediately
        - Call emergency services: 999
        - Stay on the line with emergency services

        2. **For immediate support:**
        - Go to your nearest emergency room/A&E department
        - Call The Samaritans hotline (Multilingual): (852) 2896 0000
        - Call Suicide Prevention Service hotline (Cantonese): (852) 2382 0000

        **Are you currently in a safe location?** If not, please seek immediate help using the emergency contacts above.
        ***
        """}
    ]
    
    # Add chat history
    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            follow_up_prompt.append(message)
    
    # Add current user input
    follow_up_prompt.append({"role": "user", "content": user_input})
    
    # Get response from GPT
    response = chat_with_gpt(follow_up_prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    return response

# Function to generate a diagnosis report
def generate_report():
    try:
        if not st.session_state.diagnosis["assessment_results"]:
            st.warning("No assessments have been completed yet. The report may be limited.")
        
        report_prompt = [
            {
                    "role": "system",
                    "content": """You are a mental health report specialist. Generate a comprehensive mental health diagnosis report based on the screening conversation and assessment results.

Report Structure:
1. Patient Information (extract from conversation)
2. Presenting Symptoms (summarize symptoms mentioned in conversation)
3. Assessment Results (detailed results of each assessment with scores and interpretations)
4. Diagnosis (provide a tentative diagnosis based on assessments and symptoms)
5. Recommendations (suggest appropriate treatments or further evaluations)
6. Disclaimer (include a clear and prominent disclaimer section)

Example Report:
# Mental Health Assessment Report
## Date: [Current Date]

### Patient Information
[Extracted from conversation]

### Presenting Symptoms
- [List of symptoms]
- [Duration and severity]
- [Impact on daily life]

### Assessment Results
[Detailed results of each assessment]

### Diagnosis
[Tentative diagnosis based on symptoms and assessments]

### Recommendations
[Specific recommendations for next steps]

### Disclaimer
IMPORTANT DISCLAIMER: This report is generated by an AI assistant and is not a clinical diagnosis. 
The assessment tools used are screening instruments only and do not replace a proper evaluation by a qualified healthcare professional.
This report is not a substitute for professional medical advice, diagnosis, or treatment.
If you're experiencing severe symptoms or having thoughts of harming yourself or others, please seek immediate medical attention or contact a crisis helpline."""
                }
            {
                    "role": "system",
                    "content": """You are a mental health report specialist. Generate a comprehensive mental health diagnosis report based on the screening conversation and assessment results.

Report Structure:
1. Patient Information (extract from conversation)
2. Presenting Symptoms (summarize symptoms mentioned in conversation)
3. Assessment Results (detailed results of each assessment with scores and interpretations)
4. Diagnosis (provide a tentative diagnosis based on assessments and symptoms)
5. Recommendations (suggest appropriate treatments or further evaluations)
6. Disclaimer (include a clear and prominent disclaimer section)

Example Report:
# Mental Health Assessment Report
## Date: [Current Date]

### Patient Information
[Extracted from conversation]

### Presenting Symptoms
- [List of symptoms]
- [Duration and severity]
- [Impact on daily life]

### Assessment Results
[Detailed results of each assessment]

### Diagnosis
[Tentative diagnosis based on symptoms and assessments]

### Recommendations
[Specific recommendations for next steps]

### Disclaimer
IMPORTANT DISCLAIMER: This report is generated by an AI assistant and is not a clinical diagnosis. 
The assessment tools used are screening instruments only and do not replace a proper evaluation by a qualified healthcare professional.
This report is not a substitute for professional medical advice, diagnosis, or treatment.
If you're experiencing severe symptoms or having thoughts of harming yourself or others, please seek immediate medical attention or contact a crisis helpline."""
                }
        ]
        
        # Add chat history
        for message in st.session_state.messages:
            if message["role"] in ["user", "assistant"]:
                report_prompt.append(message)
        
        # Add assessment results
        assessment_results = "Assessment Results Summary:\n"
        for assessment, result in st.session_state.diagnosis["assessment_results"].items():
            assessment_data = ASSESSMENTS[assessment]
            assessment_results += f"- {assessment_data['name']} ({assessment_data['description']})\n"
            assessment_results += f"  Score: {result['score']}\n"
            assessment_results += f"  Interpretation: {result['interpretation']}\n\n"
        
        conditions = ", ".join(st.session_state.diagnosis["possible_conditions"]) if st.session_state.diagnosis["possible_conditions"] else "No specific conditions identified"
        assessment_results += f"Possible conditions identified during screening: {conditions}\n\n"
        
        report_prompt.append({"role": "user", "content": f"Generate a comprehensive diagnosis report based on our conversation and the following assessment results:\n{assessment_results}\nInclude today's date ({datetime.now().strftime('%B %d, %Y')}) in the report header."})
        st.session_state.messages.append({"role": "assistant", "content": "report generating..."})
        report = generate_report_with_gpt(report_prompt)
        st.session_state.messages.append({"role": "assistant", "content": report})
        st.session_state.report_generated = True
        st.session_state.chat_state = "follow_up"
        
        # Add a message inviting follow-up questions
        follow_up_invitation = """I've generated your report based on our conversation and assessment results. 

You can now:
1. Ask questions about your assessment results
2. Get more information about mental health conditions
3. Discuss your concerns about the recommendations
4. Learn more about self-care strategies

What would you like to know more about?"""
        
        st.session_state.messages.append({"role": "assistant", "content": follow_up_invitation})
        st.rerun()
        return report
    except Exception as e:
        error_message = str(e)
        if "401" in error_message or "无效的令牌" in error_message:
            st.error("Authentication Error: Please check your API key. It appears to be invalid.")
            raise e
        error_message = str(e)
        if "401" in error_message or "无效的令牌" in error_message:
            st.error("Authentication Error: Please check your API key. It appears to be invalid.")
            raise e
        error_message = f"Error generating report: {str(e)}"
        st.error(error_message)
        
        fallback_report = f"""
        # Mental Health Report
        
        ## Date: {datetime.now().strftime('%B %d, %Y')}
        
        ### Assessment Results
        {assessment_results}
        
        ### Note
        There was an issue generating the complete report. Please try restarting the conversation or contact support.
        
        This report is not a substitute for professional psychiatric evaluation. Please consult with a mental health professional for a comprehensive assessment.
        """
        
        st.session_state.messages.append({"role": "assistant", "content": fallback_report})
        return fallback_report

# Initialize session state
initialize_session_state()

# Streamlit UI
st.title("Mental Health Initial Diagnosis Chatbot")

# Add an auto-scroll container for the chat
chat_container = st.container()

# Show welcome message if no messages exist
if not st.session_state.messages:
    welcome_message1 = {
        "role": "assistant", 
        "content": """Welcome to the Mental Health Chatbot.

***I'm here to help assess your mental health and provide initial diagnosis. We'll start with a conversation to understand your concerns, then I may ask you to complete one or more standardized assessments, and finally I'll provide a report summarizing our findings.***

***Please note that this is not a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing a mental health emergency, please contact emergency services or a crisis helpline immediately.***

***The conversation is confidential and will not be shared with anyone without your consent.***\n _________"""
    }
    st.session_state.messages.append(welcome_message1)
    welcome_message2 = {
        "role": "assistant", 
        "content": """Hi, i am the Mental Health Diagnosis Chatbot, how are you feeling today?"""
    }
    st.session_state.messages.append(welcome_message2)

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.messages:
        js = '''
        <script>
            function scrollToBottom() {
                const messages = document.querySelector('[data-testid="stChatMessageContainer"]');
                if (messages) {
                    messages.scrollTop = messages.scrollHeight;
                }
            }
            scrollToBottom();
        </script>
        '''
        st.components.v1.html(js, height=0)

# Input for user
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if st.session_state.chat_state == "screening":
        response = screening_agent(user_input)
    elif st.session_state.chat_state == "assessment":
        response = "I see you've sent a message during the assessment. Please use the buttons above to answer the current assessment question. If you need to stop the assessment, you can click 'Start New Conversation'."
        st.session_state.messages.append({"role": "assistant", "content": response})
    elif st.session_state.chat_state == "follow_up":
        response = follow_up_agent(user_input)
    else:
        response = "I'm not sure what to do with your message. Please try starting a new conversation."
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

# Display assessment interface if needed
if st.session_state.chat_state == "assessment" and st.session_state.current_assessment:
    assessment_agent()

# Generate Report button
if st.session_state.chat_state == "awaiting_report":
    if st.button("Generate Report"):
        st.session_state.messages.append({"role": "assistant", "content": "Generating your comprehensive report..."})
        report = generate_report()
        st.rerun()

# Reset button
if st.button("Start New Conversation"):
    st.session_state.messages = []
    st.session_state.chat_state = "screening"
    st.session_state.diagnosis = {
        "possible_conditions": [],
        "assessment_results": {},
        "final_diagnosis": "",
        "recommendations": ""
    }
    st.session_state.current_assessment = None
    st.session_state.assessment_responses = {}
    st.session_state.assessment_index = 0
    st.rerun()

# Debug info
if st.checkbox("Show Debug Info"):
    st.write(f"Current State: {st.session_state.chat_state}")
    st.write(f"Current Assessment: {st.session_state.current_assessment}")
    st.write(f"Assessment Index: {st.session_state.assessment_index}")
    st.write(f"Diagnosis Data: {st.session_state.diagnosis}")
    st.write(f"Assessment Responses: {st.session_state.assessment_responses}")
