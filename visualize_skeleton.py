import numpy as np
import matplotlib.pyplot as plt

# Esempio di tensore con shape (16, 30, 25, 3)
tensor = np.random.randn(16, 30, 25, 3)

# Itera su ogni elemento del batch
for i in range(tensor.shape[0]):
    # Crea una figura e un set di assi per il batch corrente
    fig, ax = plt.subplots()

    # Itera su ogni scheletro nella sequenza
    for j in range(tensor.shape[1]):
        # Estrai le posizioni x, y, z delle giunture per il batch e lo scheletro corrente
        x = tensor[i, j, :, 0]
        y = tensor[i, j, :, 1]
        z = tensor[i, j, :, 2]

        # Disegna le connessioni tra le giunture
        ax.plot(x, y, z, marker='o')

    # Imposta i limiti degli assi per garantire una visualizzazione corretta
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Aggiungi etichette agli assi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mostra il grafico per il batch corrente
    plt.show()
