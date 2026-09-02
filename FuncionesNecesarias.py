import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from matplotlib.patheffects import withStroke

def calcular_cramers_v_ovr(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calcula la V de Cramér One-vs-Rest (OvR) entre cada predictor binario y 
    cada clase de la variable objetivo multiclase.
    """
    diagnosticos = y.unique()
    sintomas = X.columns
    
    # Matriz para almacenar la V de Cramer de cada combinacion (sintoma vs diagnostico)
    resultados_ovr = pd.DataFrame(index = sintomas, columns = diagnosticos, dtype=float)
    
    for diagnostico in diagnosticos:
        # Binarizamos la variable objetivo: 1 si es la diagnostico actual, 0 para el resto
        y_ovr = (y == diagnostico).astype(int)
        
        for sintoma in sintomas:
            # Crear la tabla de contingencia 2x2 (sintoma vs diagnostico_OVR)
            tabla_contingencia = pd.crosstab(X[sintoma], y_ovr)
            
            # Prueba de Chi-Cuadrado para consistencia
            chi2, _, _, _ = chi2_contingency(tabla_contingencia, correction=False)
            n = tabla_contingencia.sum().sum()
            
            # Formulavde V de Cramer para tabla 2x2
            v_cramer = np.sqrt(chi2 / n)
            
            resultados_ovr.loc[sintoma, diagnostico] = v_cramer

    # Crear un DataFrame de resumen con la V de Cramer maxima por sintoma
    resumen = pd.DataFrame({
        'v_cramer_max': resultados_ovr.max(axis = 1),
        'diagnostico_asociado': resultados_ovr.idxmax(axis = 1)
    }).sort_values(by='v_cramer_max', ascending=False)
    
    return resultados_ovr, resumen

def analizar_y_graficar_sintoma(
    sintoma: str, 
    df_input: pd.DataFrame, 
    variable_objetivo: pd.Series, 
    umbral: float = 2.0,
    ancho: float = 10.0,
    alto: float = 8.0,
    tamano_letra: float = 7.0,
    print_output: bool = True
):
    """
    Grafica la distribución porcentual filtrando los casos minoritarios (< umbral) 
    sin crear la categoría 'Otros', y genera un reporte de coincidencias en texto.
    """
    # 1. Crear tabla cruzada con conteos reales y proporciones
    ct_counts = pd.crosstab(df_input[sintoma], variable_objetivo)
    ct_prop = pd.crosstab(df_input[sintoma], variable_objetivo, normalize='index') * 100

    # 2. Filtrar únicamente las categorías que superan el umbral
    ct_procesada = ct_prop.copy()
    ct_procesada[ct_procesada < umbral] = 0.0

    # Eliminar columnas que se quedaron en 0 para ambas barras
    cols_visibles = ct_procesada.columns[(ct_procesada > 0).any(axis=0)]
    ct_procesada = ct_procesada[cols_visibles]

    # 3. Configurar la figura y graficar
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ct_procesada.plot(
        kind='bar', stacked=True, ax=ax, colormap='tab20', 
        edgecolor='white', linewidth=0.5, width=0.45, legend=False
    )

    # 4. Anotar porcentajes solo en los bloques dibujados
    for i, col in enumerate(ct_procesada.columns):
        for j, p in enumerate(ax.patches[i*len(ct_procesada) : (i+1)*len(ct_procesada)]):
            height = p.get_height()
            if height >= umbral:
                x_center = p.get_x() + p.get_width() / 2.
                y_center = p.get_y() + height / 2.
                
                font_size = tamano_letra + 2.0 if height >= 8.0 else max(tamano_letra - 0.5, 5.0)
                nombre_corto = col if len(col) <= 18 else f"{col[:15]}.."
                texto = f"{col}\n({height:.1f}%)" if height >= 8.0 else f"{nombre_corto} ({height:.1f}%)"

                ax.annotate(
                    texto, (x_center, y_center), ha='center', va='center',
                    fontsize=font_size, color='white', fontweight='bold', clip_on=False,
                    path_effects=[withStroke(linewidth=0.8, foreground='black')]
                )

    ax.set_title(f"Distribución de diagnósticos según '{sintoma}'", fontsize=tamano_letra + 5.0, pad=15, fontweight='bold')
    ax.set_xlabel(f"Estado del Síntoma '{sintoma}'", fontsize=tamano_letra + 3.0, labelpad=8)
    ax.set_ylabel("Porcentaje de Pacientes (%)", fontsize=tamano_letra + 3.0)
    ax.set_xticklabels(['Ausente (0)', 'Presente (1)'], rotation=0, fontsize=tamano_letra + 3.0, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 1.6)

    plt.tight_layout()
    plt.show()

    # 5. REPORTE EN TEXTO DE COINCIDENCIAS (Presente vs Ausente)
    if print_output == True:
        if 1 in ct_counts.index and 0 in ct_counts.index:
            diag_presente = ct_counts.loc[1][ct_counts.loc[1] > 0].index.tolist()
            
            coincidencias = []
            for diag in diag_presente:
                pacientes_ausente = ct_counts.loc[0, diag]
                pct_ausente = ct_prop.loc[0, diag]
                if pacientes_ausente > 0:
                    coincidencias.append((diag, pacientes_ausente, pct_ausente))

            print("="*70)
            print(f"Diagnósticos principales en 'Presente (1)': {len(diag_presente)}")
            
            if coincidencias:
                print(f"\n Condiciones del lado derecho que TAMBIÉN aparecen con síntoma 'Ausente (0)':\n")
                for diag, cant, pct in coincidencias:
                    print(f"   • {diag}: {cant} paciente(s) ({pct:.2f}% de la barra Ausente)")
            else:
                print(f"\n Ninguna de las condiciones del lado derecho aparece cuando '{sintoma}' está Ausente (0).")
            print("="*70 + "\n")
    
def seleccionar_top_sintomas_por_clase(matriz_ovr: pd.DataFrame, top_n_por_clase: int = 2) -> list:
    """
    Selecciona los N síntomas con mayor V de Cramér para CADA clase (diagnostico)
    y devuelve la lista única de síntomas resultantes sin duplicados.
    
    Parámetros:
    -----------
    matriz_ovr : pd.DataFrame
        DataFrame de dimensión (síntomas x diagnosticos) generado en el paso anterior.
    top_n_por_clase : int (por defecto 2)
        Número de síntomas con mayor V de Cramér a conservar por cada diagnostico.
    """
    sintomas_seleccionados = set()
    
    # Recorremos cada columna
    for diagnostico in matriz_ovr.columns:
        # Ordenamos los sintomas de mayor a menor V de Cramer
        top_sintomas = matriz_ovr[diagnostico].nlargest(top_n_por_clase).index.tolist()
        
        # Agregamos al conjunto
        sintomas_seleccionados.update(top_sintomas)
        
    sintomas_finales = sorted(list(sintomas_seleccionados))
    return sintomas_finales

def mostrar_tabla_top_sintomas(matriz_ovr: pd.DataFrame, top_n: int = 2):
    """
    Construye y grafica una tabla limpia con los N síntomas top y sus valores de V de Cramér 
    para cada diagnostico en la matriz OvR.
    """
    filas = []
    
    # Extraer los datos ordenados por cada diagnostico
    for diagnostico in matriz_ovr.columns:
        sintomas_top = matriz_ovr[diagnostico].nlargest(top_n)
        for i, (sintoma, val_cramer) in enumerate(sintomas_top.items(), 1):
            filas.append({
                'Diagnostico': diagnostico,
                'Jerarquía': f"Top {i}",
                'Síntoma Clave': sintoma,
                'V de Cramér (OvR)': f"{val_cramer:.4f}"
            })
            
    df_tabla = pd.DataFrame(filas)
    return df_tabla

def motor_triaje_interactivo(modelo, lista_sintomas, sintomas_iniciales, umbral_certeza=0.85, max_preguntas=8):
    sintomas_validos = [s for s in sintomas_iniciales if s in lista_sintomas]
    
    if not sintomas_validos:
        print("Ninguno de los síntomas ingresados está en el conjunto de 77 variables.")
        return

    # Vector del paciente
    estado_paciente = pd.DataFrame([np.zeros(len(lista_sintomas))], columns = lista_sintomas)
    
    # Activar sintomas iniciales
    for s in sintomas_validos:
        estado_paciente.at[0, s] = 1
        
    sintomas_evaluados = set(sintomas_validos)
    preguntas_realizadas = 0
    
    print("=== CONSULTA ADAPTATIVA INTERACTIVA ===")
    print(f"Síntomas iniciales presentes: {sintomas_validos}\n")
    
    while preguntas_realizadas < max_preguntas:
        # Calcular probabilidades a posteriori
        probs = modelo.predict_proba(estado_paciente)[0]
        clases = modelo.classes_
        
        idx_top = np.argmax(probs)
        patologia_top = clases[idx_top]
        certeza_actual = probs[idx_top]
        
        print(f"[Estado actual] Diagnóstico probable: {patologia_top} ({certeza_actual * 100:.2f}%)")
        
        if certeza_actual >= umbral_certeza:
            print(f"\n ¡Umbral de certeza alcanzado ({certeza_actual * 100:.2f}% >= {umbral_certeza * 100:.0f}%)!")
            break
            
        # Ranking de importancia sobre las clases sospechosas (Top 3)
        top_indices = np.argsort(probs)[::-1][:3]
        coefs_top = np.abs(modelo.coef_[top_indices])
        peso_sintomas = coefs_top.sum(axis = 0)
        
        df_importancia = pd.DataFrame({'sintoma': lista_sintomas, 'peso': peso_sintomas}).sort_values(by = 'peso', ascending=False)
        candidatos = df_importancia[~df_importancia['sintoma'].isin(sintomas_evaluados)]
        
        if candidatos.empty:
            print("\nSe han evaluado todos los síntomas disponibles.")
            break
            
        sintoma_sugerido = candidatos.iloc[0]['sintoma']
        sintomas_evaluados.add(sintoma_sugerido)
        preguntas_realizadas += 1
        
        # Entrada del usuario por consola (1 para SI, 0 para NO)
        respuesta = input(f"Pregunta #{preguntas_realizadas}: ¿El paciente presenta '{sintoma_sugerido}'? (1 = SÍ / 0 = NO): ").strip()
        
        if respuesta == '1':
            estado_paciente.at[0, sintoma_sugerido] = 1
            print(f"   [+] '{sintoma_sugerido}' marcado como PRESENTE.")
        else:
            estado_paciente.at[0, sintoma_sugerido] = 0
            print(f"   [-] '{sintoma_sugerido}' descartado.")
            
        print("-" * 60)

    print("\n=== DIAGNÓSTICO FINAL ===")
    print(f"Patología: {patologia_top}")
    print(f"Certeza final: {certeza_actual * 100:.2f}%")
    print(f"Preguntas realizadas: {preguntas_realizadas}")
    
def obtener_sintomas_por_diagnostico(df, diagnostico, umbral_frecuencia = 0.0):
    """
    Retorna la lista de síntomas asociados a un diagnóstico específico.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame con los datos (debe contener la columna 'prognosis').
    diagnostico : str
        El nombre exacto de la enfermedad (ej: 'Fungal infection', 'Diabetes ', etc.).
    umbral_frecuencia : float, opcional (por defecto 0.0)
        Porcentaje mínimo (de 0.0 a 1.0) en que debe aparecer el síntoma en los pacientes 
        con dicha enfermedad. Usar 0.5 para síntomas presentes en al menos el 50% de los casos.
        
    Retorna:
    --------
    list : Lista con los nombres de los síntomas.
    """
    # Filtrar los datos para el diagnostico solicitado
    df_enfermedad = df[df['prognosis'] == diagnostico]
    
    if df_enfermedad.empty:
        print(f"El diagnóstico '{diagnostico}' no se encuentra en el dataset.")
        return []
    
    # Seleccionar solo las columnas de sintomas
    columnas_sintomas = [col for col in df.columns if col != 'prognosis']
    
    # Calcular la frecuencia relativa de cada sintoma
    frecuencias = df_enfermedad[columnas_sintomas].mean()
    
    # Filtrar los sintomas que superen el umbral especificado
    sintomas_presentes = frecuencias[frecuencias > umbral_frecuencia].sort_values(ascending = False)
    
    return list(sintomas_presentes.index)