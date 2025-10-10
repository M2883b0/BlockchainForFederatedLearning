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
    ["./client_1_model.pth", "./client_2_model.pth"]
  ],
  "votes": [
    ["client_1", "client_2"]
  ],
  "contribution": {
    "client_1": 0.5
  }
}
```
