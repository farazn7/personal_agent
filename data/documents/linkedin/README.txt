File 1: post_erc_technical.txt

Topic: ERC System Architecture

Securing 6th rank at ERC Remote 2025 wasn't just about driving; it was about building a reliable remote autonomy stack from scratch.

As the Software Lead, I focused on orchestrating a modular pipeline that could handle high-latency telemetry. Key technical wins:

    Navigation: Implemented a custom Nav2 stack tuned for rugged terrain, moving beyond default costmap behaviors to handle dynamic obstacle inflation.

    Sensor Fusion: Optimized EKF (Robot Localization) by fusing IMU and wheel odometry to mitigate slip-drift during precise maneuvers.

    Communication: Designed a custom ROS2-based bridge to ensure command persistence despite low-bandwidth remote links.

This was the first fully autonomous-capable rover from the Mars Club, and the leap from "teleoperated" to "goal-oriented autonomy" was the real victory.

#ROS2 #Nav2 #RoboticsEngineering #AutonomousSystems #ERC2025
File 2: post_irc_slam_focus.txt

Topic: SLAM and Drive Subsystems

Reflecting on our 10th rank at the International Rover Challenge (IRC). While the rank is great, the engineering behind our drive subsystem was the highlight for me.

I spent most of my time in the "trenches" of SLAM and sensor fusion. We integrated Lidar-based SLAM with a custom-tuned controller to ensure the rover didn't just move, but understood its environment in real-time.

The Challenge: Balancing the compute load on our onboard hardware while maintaining a high-frequency control loop for the drive subsystem.
The Result: A fail-safe modular architecture where the navigation stack could recover gracefully from sensor dropouts—a first for our club's software heritage.

#SLAM #Robotics #Linux #EmbeddedSystems #MarsRover
File 3: post_mlops_assistant.txt

Topic: Assistant Architecture (The "Why" and "How")

(Keeping the "Beautiful" vibe of the pgvector post but adding more technical depth.)

Why am I building a Generator-Evaluator-Optimizer loop instead of a simple RAG chain?

Standard RAG is brittle. For my Personal Assistant, I needed a system that doesn't just "search and retrieve" but "reasons and refines."

    The Router: A lightweight Llama 3.2 3B model that classifies intent—deciding if the query needs a PGVector lookup or a Tavily web search.

    The Evaluator: A Qwen 2.5 7B agent that critiques the initial draft against my "Style Matcher" files.

    The Storage: Leveraging psycopg[pool] for efficient connection management to a PGVector-enabled Postgres instance.

Architecture is the difference between a toy and a tool.

#MLOps #LangGraph #PostgreSQL #LLM #SoftwareArchitecture