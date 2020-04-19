### How to Install


```
pip install -r requirements.txt
```

### CartPole enviroment

O objeto retornado de cada iteração no ambiente é chamado _observação_, essa observação contem as informações relevantes para o treinamento. No ```CartPole-v0``` a observação é a seguinte:

obs = [cP, cS, pA, pS]

sendo:
* cP: posição do carro, varia de -4.8 a 4.8
* cS: velocidade do carro, varia de -5 a 5
* pA: angulo do poste, varia de -0.418 a 0.418
* pS: velocidade do poste, varia de -5 a 5

Dado as circunstancias do ambiente tanto a velocidade quanto a posição tem apenas uma direção (valores negativos esquerda valores positivos direita).
