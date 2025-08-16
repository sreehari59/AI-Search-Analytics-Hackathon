# AI Search Analytics Hackathon - Consumer Behaviour on LLMs

## 🏆 Project Overview
**Consumer behaviour on LLMs - AI Search Analytics Hackathon (€10k winner)**

This project was built during the AI Search Hackathon organized by Peec.ai, where we placed **1st out of 36 teams** and won the **€10,000 grand prize**.

## 🚀 What We Built
**A sophisticated Chrome extension** that creates a **win-win ecosystem** between users and brands in AI Search. The extension seamlessly integrates with ChatGPT and other AI interfaces, where users receive valuable discount coupons in exchange for sharing their AI interaction data, giving brands unprecedented visibility into consumer behavior on AI platforms.

## 🔧 Chrome Extension - The Core Innovation

### **Business Model: Data-for-Value Exchange**
- **User Incentive**: Receive relevant discount coupons and product recommendations
- **Brand Value**: Gain real-time insights into how consumers interact with AI
- **Data Transparency**: Users knowingly share their AI conversations in exchange for savings
- **Privacy-First**: Ethical data collection with clear value exchange

### **Technical Architecture**
- **Manifest V3**: Built with the latest Chrome extension standards
- **Content Scripts**: Automatically injects into ChatGPT conversations
- **Real-time Monitoring**: Captures every user prompt and AI response
- **Product Injection**: Dynamically displays relevant product links and coupons
- **Cross-platform**: Works on chat.openai.com and chatgpt.com

### **How It Works**
1. **User Interaction Detection**: Monitors ChatGPT conversations in real-time
2. **Prompt Capture**: Extracts user questions and AI responses automatically
3. **Intent Analysis**: Processes prompts to understand user behavior
4. **Product Matching**: Identifies relevant products and discount opportunities
5. **Seamless Integration**: Injects product cards directly into the chat interface
6. **Value Exchange**: Users get coupons, brands get consumer insights

### **Key Features**
- **Smart Crawling**: Automatically detects conversation turns and extracts content
- **Dynamic UI**: Beautiful, responsive product cards with hover effects
- **Real-time Processing**: Sends data to backend for instant analysis
- **User Experience**: Non-intrusive integration that enhances rather than disrupts
- **Incentive System**: Discount coupons as reward for data sharing

## 🔍 Analytics & Intelligence
- **User Prompt Analysis**: Captures and processes user interactions with LLMs
- **Intent Classification**: Classifies user intent into three categories:
  - **Consideration**: Initial research and exploration
  - **Evaluation**: Comparing options and features
  - **Decision**: Ready to make a purchase decision
- **Prompt Clustering**: Groups similar prompts to identify patterns and trends
- **Brand Insights**: Generates actionable recommendations for brands and content creators

## 🛠️ Technical Implementation

### **Frontend (Chrome Extension)**
- **Content Scripts**: JavaScript injection for real-time monitoring
- **DOM Manipulation**: Dynamic product card creation and injection
- **Event Handling**: Keyboard and conversation state monitoring
- **Responsive Design**: Grid-based layout with hover animations

### **Backend & Analytics**
- **Data Generation**: Scripts to create realistic training data for LLM interactions
- **K-Means Clustering**: Groups user prompts by similarity using TF-IDF vectorization
- **Intent Modeling**: Machine learning models to classify user behavior patterns
- **Analytics Pipeline**: Processes and analyzes user interaction data

### **Data Processing**
- **Real-time Streaming**: Live data capture from ChatGPT conversations
- **API Integration**: Seamless communication with backend services
- **Data Storage**: Structured CSV output for analysis and insights

## 📊 Data Structure
The project generates and analyzes:
- **Chat Data**: User prompts and AI responses with timestamps
- **Action Data**: User interactions (clicks, purchases) linked to content
- **Clustered Results**: Grouped prompts with identified patterns

## 🎯 Business Impact
This solution creates a **revolutionary data ecosystem** that provides brands with:

- **AI Consumer Intelligence**: Unprecedented visibility into how users interact with AI platforms
- **Real-time Consumer Insights**: Live monitoring of user behavior on AI platforms
- **Purchase Journey Mapping**: Understanding of user intent throughout the decision process
- **Data-driven Recommendations**: AI-powered content and product suggestions
- **Competitive Advantage**: First-mover advantage in the emerging AI Search ecosystem
- **Revenue Optimization**: Direct integration of discount coupons and product links
- **Ethical Data Collection**: Transparent value exchange model that respects user privacy
- **Market Research Goldmine**: Access to real consumer conversations and decision-making processes

## ⚠️ Technical Notes
*Due to hackathon time constraints, we implemented simplified machine learning models (K-means clustering with TF-IDF) rather than more complex deep learning approaches. The models focus on prompt similarity and basic intent classification, providing a solid foundation for future enhancements.*

## 🏅 Achievement
**🏆 1st Place Winner** - AI Search Hackathon by Peec.ai  
**💰 €10,000 Grand Prize**  
**👥 36 Competing Teams**

---

*Built during the AI Search Analytics Hackathon - A project exploring the future of consumer behavior analysis in AI-powered search environments through innovative Chrome extension technology.*