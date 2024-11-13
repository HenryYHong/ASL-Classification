\documentclass{article}
\usepackage{hyperref}

\title{ASL Classification - Real-time American Sign Language Recognition}
\begin{document}
\maketitle

\section*{Project Overview}
This project aims to develop a computer vision program for recognizing American Sign Language (ASL) hand gestures in real time. Using a machine learning model trained on an MNIST-style dataset with ASL data, the program interprets hand signs and classifies them into corresponding ASL letters or words.

\section*{Features}
\begin{itemize}
    \item \textbf{Image Preprocessing}: Efficient image preprocessing pipeline to normalize and prepare gesture images.
    \item \textbf{Model Architecture}: Leverages Convolutional Neural Networks (CNNs) to capture spatial features unique to each hand gesture.
    \item \textbf{Real-time Recognition}: Processes live video feed for real-time ASL recognition.
    \item \textbf{User Interface}: Interactive interface to display recognized ASL letters in real-time.
\end{itemize}

\section*{Tech Stack}
Python, OpenCV, TensorFlow, and Keras

\section*{Usage}
Clone the repository, install dependencies, and run the script to start the ASL recognition program on your local machine.

\section*{Future Enhancements}
\begin{itemize}
    \item Support for full words and phrases
    \item Addition of more ASL gestures
    \item Optimization for mobile platforms
\end{itemize}

\end{document}
