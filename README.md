# 🚀 AI_Content_Scanner

**AI_Content_Scanner** is a powerful framework built on **NVIDIA Inference Microservices (NIM)**, designed to streamline the deployment and scaling of Generative AI models. It delivers optimized performance for LLMs and other generative applications with robust monitoring, scaling, and performance management tools.

---

## 🧠 Key Highlights

- 🔗 Seamless integration with NVIDIA NIM for model deployment and orchestration
- ⚙️ Dynamic batching and model parallelism for scalable, high-efficiency inference
- 📈 Real-time metrics and intelligent auto-scaling based on usage
- 📊 Insights into latency, throughput, and GPU utilization
- 🖥️ Interactive dashboard for managing models and monitoring performance

---

## 📋 Requirements

Before getting started, ensure your system meets the following:

- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/)
- [Helm](https://helm.sh/)
- NVIDIA GPU with CUDA support
- Python 3.8 or higher

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/talibhussain5545/AI_Content_Scanner.git
   cd AI_Content_Scanner
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your environment**
   Modify the configuration file to match your setup:

   ```
   config/config.yaml
   ```

---

## 🚀 Getting Started

1. **Launch the accelerator service**

   ```bash
   python src/main.py
   ```

2. **Open the web dashboard**

   ```bash
   streamlit run ui/dashboard.py
   ```

3. **Use the provided APIs** for deploying models, making inferences, and managing workloads.

---

## 🧪 Testing

To execute all unit tests:

```bash
python -m unittest discover tests
```

---

## 📊 Performance Benchmarking

Run benchmark tests to measure model performance:

```bash
python benchmarks/benchmark.py
```

---

## ☁️ Deployment Options

For deploying AI_Content_Scanner in a production-grade environment, use Kubernetes or Helm:

- **Kubernetes**

  ```bash
  kubectl apply -f kubernetes/ -n genai-accelerator
  ```

- **Helm**

  ```bash
  helm install genai-accelerator helm/
  ```

---
