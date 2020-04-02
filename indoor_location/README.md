Pretrain Bert 
    epochs = 100  
    learning_rate=1e-3,
    decay_steps=30000,
    warmup_steps=10000,
    weight_decay=1e-3,
    loss最好到：（降不下去了）
    (跑了一夜才跑了50多个epoch)
    loss: 1.0513 - MLM_loss: 1.0493 - NSP_loss: 0.0020 
    - val_loss: 0.9864 - val_MLM_loss: 0.9847 - val_NSP_loss: 0.0017