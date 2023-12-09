import cv2
import numpy as np

def obtener_dados(img):
	conComps = cv2.connectedComponentsWithStats(img, 8)
	
	# Verificamos posicion de las componentes conectadas
	dados = []
	for stat in conComps[2]: # Filtramos para quedarnos sólo con dados
		if stat[2] < stat[3]*1.5 and stat[2]*1.5 > stat[3] and stat[2]*stat[3] > 500 and stat[2]*stat[3] < 1300: 
			dados.append(stat)
	return dados

def obtener_quietos(dadosNuevos, dadosViejos):
	# Comparamos la posición relativa de los dados
	quietos = []
	for stat_new in dadosNuevos:
		for stat_old in dadosViejos:
			# Vemos si tiene algún dado suficientemente cerca, lo que significaría que está quieto
			if abs(stat_old[0]-stat_new[0]) < 2 and abs(stat_old[1]-stat_new[1]) < 2: 
				quietos.append(stat_new)
	return quietos


def valores(dados, frame):
	# Inicializamos valorDados como -1 porque la primer componente conectada será la correspondiente
	# al fondo, y no quremos contarla.
	valorDados = [-1] * len(dados)
	j = 0
	for st in dados:
		framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subimg = framegray[st[1]:st[1]+st[3], st[0]:st[0]+st[2]]

		# Umbralamos la imagen
		_, binSubimg = cv2.threshold(subimg, 175, 255, cv2.THRESH_BINARY)

		_, _, stats, _ = cv2.connectedComponentsWithStats(binSubimg, 8)
		
		# Filtramos para que un pixel suelto no sea considerado para el valor del dado
		for stat in stats:
			if stat[4] > 3: # Descartamos posibles puntos brillantes
				valorDados[j] +=1
		j += 1
	return valorDados

# --- Leer un video ------------------------------------------------
filename = input("Ingrese el nombre del archivo a analizar: ")
cap = cv2.VideoCapture(filename)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

dadosOld = []
dadosNew = []
out = cv2.VideoWriter(f'out_{filename}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width/3),int(height/3)))
while (cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
		# Vemos si la imagen varió respecto del frame anterior
		img_blur = cv2.GaussianBlur(frame, (3,3), 2)
		img_canny = cv2.Canny(img_blur, 150, 250, apertureSize=3, L2gradient=True) 

		kernel_size = 4
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		dilated = cv2.dilate(img_canny, kernel, iterations=1)
		
		dadosNew = obtener_dados(dilated)
			
		# Si es la primera vez que encontramos un dado, probablemente esté muy en movimiento,
		# no queremos mostrarlo hasta asegurarnos de que esté quieto.
		if dadosOld == []:
			dadosOld = dadosNew
			out.write(frame)
			continue
		
		# Obtenemos cuáles de los dados están en reposo
		dadosQuietos = obtener_quietos(dadosOld, dadosNew)
		# Guardamos todos los dados antes de solo tomar los válidos
		dadosOld = dadosNew
		# Ahora, dibujamos las bounding boxes de los dados en la imagen
		for st in dadosQuietos:
			_ = cv2.rectangle(frame, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(255,0,0), thickness=2)

		# Parte 2: Contamos el valor de cada dado.
		valoresDados = valores(dadosQuietos, frame)

		# Escribimos el valor de los dados arriba de ellos.
		for i in range(len(dadosQuietos)):
			_ = cv2.putText(frame, str(valoresDados[i]), (dadosQuietos[i][0],dadosQuietos[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

		# Grabamos el resultado
		out.write(frame)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break
cap.release()
