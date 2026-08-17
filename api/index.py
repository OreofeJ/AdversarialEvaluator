import asyncio
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Adversarial Robustness API")

# Shared process pool executor across warm function invocations
executor = ProcessPoolExecutor(max_workers=2)

class EvalRequest(BaseModel):
    model_id: str
    epsilon: float
    samples: Optional[List[List[float]]] = None

def _run_art_fgsm(epsilon: float) -> dict:
    """Isolated execution wrapper for ART FastGradientMethod computation."""
    import numpy as np
    import torch
    import torch.nn as nn
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod

    # 1. Define dummy target model for evaluation
    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)

    model = SmallNet()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(10,),
        nb_classes=2
    )

    # 2. Synthetic sample input
    x_clean = np.random.rand(5, 10).astype(np.float32)
    y_clean = np.array([0, 1, 0, 1, 0])

    # 3. Apply FGSM Attack via ART
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_adv = attack.generate(x=x_clean)

    # 4. Calculate degradation metrics
    clean_preds = np.argmax(classifier.predict(x_clean), axis=1)
    adv_preds = np.argmax(classifier.predict(x_adv), axis=1)

    clean_acc = float(np.mean(clean_preds == y_clean))
    adv_acc = float(np.mean(adv_preds == y_clean))

    return {
        "epsilon": epsilon,
        "clean_accuracy": round(clean_acc * 100, 2),
        "robust_accuracy": round(adv_acc * 100, 2),
        "accuracy_drop": round((clean_acc - adv_acc) * 100, 2),
        "perturbed_samples_count": len(x_adv)
    }

@app.post("/api/evaluate")
async def evaluate_model(payload: EvalRequest):
    if payload.epsilon < 0.0 or payload.epsilon > 1.0:
        raise HTTPException(status_code=400, detail="Epsilon parameter must be between 0.0 and 1.0")

    loop = asyncio.get_running_loop()
    
    # Offload heavy ML matrix math to dedicated process pool executor
    result = await loop.run_in_executor(executor, _run_art_fgsm, payload.epsilon)
    
    return {"status": "success", "data": result}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Adversarial Robustness Evaluator"}