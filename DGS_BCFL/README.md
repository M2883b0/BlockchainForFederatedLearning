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
    [("learner_1_sign", "./client_1_model.pth"), "./client_2_model.pth"]
  ],
  "votes": [
    [("learner_1", "client_1", True), ("learner_1", "client_1", False)]
  ],
  "contribution": {
    "client_1": 0.5
  },
  "global_accuracy_history": [],
  "contribution_history": []
}
```
