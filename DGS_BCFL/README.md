# 
- 总字典
  - 角色字典
  - 模型字典
```python
main_dict = {
  "role": [
    {"client_1": "leaner"}
  ],
  "global_model": [
    "./global_model.pth",
  ],
  "client_gradients":[
    [("learner_1_sign", "./client_1_model.pth", 1234, 5), ]
  ],
  "votes": [
    [("learner_1", "client_1", True, 1234, 5), ("learner_1", "client_2", False, 1234, 5)]
  ],
  "contribution": {
    "client_1": 0.5
  },
  "active_clients": [],
  "suspicious_clients": [],
  "deactivate_clients": [],
  "global_accuracy_history": [],
  "contribution_history": []
}
```
