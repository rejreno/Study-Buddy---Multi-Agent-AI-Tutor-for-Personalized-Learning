# Study Buddy – Multi-Agent AI Tutor

Study Buddy is a concierge-style multi-agent AI system that helps individual learners plan their studies, generate learning materials, test their knowledge, and receive structured feedback. The project demonstrates how multiple AI agents collaborate to solve a complex educational workflow.

Study Buddy automates the entire learning loop: planning, content creation, assessment, and evaluation, reducing manual effort while improving structure and consistency in self-learning.

## Problem Statement

Self-learning is often inefficient, fragmented, and time-consuming. Learners must manually decide what to study, find explanations, verify understanding, and assess progress. This process lacks structure and timely feedback, which leads to slower learning and lower motivation. Automating this workflow with AI agents addresses a real and relevant problem in education.

## Why Agents

This problem naturally decomposes into multiple interdependent tasks that benefit from specialization. A multi-agent system enables each agent to focus on a single responsibility while collaborating toward a shared goal. Agents communicate through structured messages and operate in both sequential and parallel modes, coordinated by an orchestrator agent. This approach mirrors real collaborative systems and allows better scalability, modularity, and reasoning compared to a single monolithic model.

## Architecture Overview

The system is organized around a central Orchestrator Agent that coordinates agent interactions. The Planner Agent generates a personalized study plan, the Content Generation Agent produces concise learning notes, the Quiz Master Agent creates quizzes based on generated content, and the Evaluator Agent assesses learner responses and provides feedback. Sessions, long-term memory storage, custom tools, and observability mechanisms support the agent workflow.

User input flows through the orchestrator, which dispatches tasks to specialized agents and aggregates the results into a final study artifact.

## Features Demonstrated

- Multi-agent system with specialized roles  
- Sequential and parallel agent execution  
- LLM-powered agents  
- Agent-to-Agent communication  
- Custom tools for persistence  
- Session and state management  
- Long-term memory (memory bank)  
- Context compaction  
- Observability via logging and metrics  
- Simple agent evaluation

## How to Run

Install dependencies:


