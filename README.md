# 📊 Optimizador Profesional de Portafolios de Acciones

Un sistema sofisticado de optimización de portafolios que implementa la teoría moderna de portafolios con capacidades avanzadas de backtesting y un conjunto completo de visualizaciones profesionales.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-estable-brightgreen.svg)

## 🎯 Descripción General

Este sistema implementa un optimizador de portafolios de nivel institucional que utiliza datos históricos de Yahoo Finance para construir, optimizar y evaluar estrategias de inversión cuantitativas. El código incluye correcciones críticas a problemas comunes de data leakage y bias de selección, proporcionando resultados confiables para análisis de inversión.

## 🔧 Arquitectura del Sistema

### 📁 Estructura Principal
```
professional_portfolio_optimizer_fixed.py (2,200+ líneas)
├── Configuración y Parámetros
├── Utilidades y Filtros de Calidad
├── Obtención y Limpieza de Datos
├── Cálculo de Métricas Financieras
├── Screening y Selección de Activos
├── Optimización de Portafolios
├── Framework de Backtesting
├── Análisis de Performance
├── Sistema de Visualizaciones
└── Función Principal de Ejecución
```

## ⚙️ Configuración del Sistema

### Parámetros Principales (Clase `PortfolioConfig`)
```python
# Universo de Activos
TICKERS: 25 acciones blue-chip (AAPL, MSFT, NVDA, etc.)
BENCHMARK: "^GSPC" (S&P 500)

# Parámetros de Datos
YEARS: 5 años de historia
INTERVAL: "1d" (datos diarios)
MIN_COVERAGE: 80% cobertura mínima de datos
MIN_SESSIONS: 400 sesiones mínimas en ventana rodante

# Optimización
TOP_N: 15 activos seleccionados
WEIGHT_CAP: 12.5% peso máximo por activo
RF: 2% tasa libre de riesgo anual
MIN_POSITIONS: 8 posiciones mínimas activas

# Backtesting
WINDOW_YEARS: 2 años ventana de entrenamiento
REBALANCE: "M" rebalanceo mensual
TRANSACTION_COST: 0.15% costo por transacción
MIN_WARMUP_SESSIONS: 60 sesiones mínimas antes de iniciar
```

## 🧮 Metodología y Fórmulas

### 1. **Cálculo de Métricas Financieras**

#### Métricas de Retorno
```python
# CAGR (Compound Annual Growth Rate)
CAGR = (Precio_Final / Precio_Inicial) ^ (1/años) - 1

# Retorno Anualizado
Retorno_Anual = Media_Retornos_Diarios * 252

# Retorno Total
Retorno_Total = (Precio_Final / Precio_Inicial) - 1
```

#### Métricas de Riesgo
```python
# Volatilidad Anualizada
Volatilidad = Desviación_Estándar_Diaria * sqrt(252)

# Sharpe Ratio
Sharpe = (Retorno_Anual - Tasa_Libre_Riesgo) / Volatilidad

# Maximum Drawdown
Drawdown = (Precio - Máximo_Histórico) / Máximo_Histórico
Max_Drawdown = min(Drawdown)

# VaR 95%
VaR_95 = Percentil_5_Retornos * sqrt(252)
```

#### Factores de Momentum
```python
# Momentum 6 meses
Momentum_6M = (Precio_Actual / Precio_126_días_atrás) - 1

# Momentum 12 meses  
Momentum_12M = (Precio_Actual / Precio_252_días_atrás) - 1
```

### 2. **Sistema de Screening de Activos**

#### Método "Enhanced Composite" (por defecto)
```python
Score = 0.7 * Sharpe_Normalizado + 
        0.2 * (1 - Volatilidad_Normalizada) + 
        0.1 * Momentum_Normalizado
```

#### Otros Métodos Disponibles
- **momentum_focused**: 60% Sharpe + 40% Momentum
- **risk_adjusted**: 100% Sharpe Ratio
- **low_risk**: 60% Low Volatility + 40% Low Drawdown

### 3. **Algoritmos de Optimización**

#### A. Max Sharpe Portfolio
**Objetivo**: Maximizar ratio Sharpe del portafolio
```python
Función_Objetivo = -Sharpe_Portafolio
donde Sharpe_Portafolio = (Retorno_Portafolio - RF) / Volatilidad_Portafolio

Restricciones:
- Suma de pesos = 1
- 0 ≤ peso_i ≤ 12.5% (ajustado dinámicamente)
- Mínimo 8 posiciones activas
```

#### B. Minimum Variance Portfolio
**Objetivo**: Minimizar varianza del portafolio
```python
Función_Objetivo = w^T * Σ * w
donde:
- w = vector de pesos
- Σ = matriz de covarianza anualizada

Restricciones: Igual que Max Sharpe
```

#### C. Risk Parity Portfolio
**Objetivo**: Igualar contribución de riesgo de cada activo
```python
Función_Objetivo = Σ(Contribución_Riesgo_i - Riesgo_Objetivo)²

donde:
Contribución_Riesgo_i = peso_i * (Σ * w)_i / Volatilidad_Portafolio
Riesgo_Objetivo = Volatilidad_Portafolio / N
```

### 4. **Framework de Backtesting**

#### Proceso de Rolling Window
1. **Definir ventana histórica**: 2 años de datos hasta fecha t
2. **Aplicar filtros de calidad**: Cobertura mínima 80%, sesiones suficientes
3. **Calcular métricas**: Solo con datos in-sample
4. **Screening de activos**: Seleccionar top 15 por score
5. **Optimizar portafolio**: Aplicar algoritmo elegido
6. **Calcular turnover**: Cambios vs período anterior
7. **Aplicar pesos**: Período siguiente (out-of-sample)
8. **Descontar costos**: 15 bps * turnover en primer día

#### Correcciones Críticas Implementadas
- ✅ **Eliminado data leakage**: No clip_outliers en datos OOS
- ✅ **Lógica de warm-up mejorada**: Mínimo 60 sesiones antes de iniciar
- ✅ **Conflicto peso máximo resuelto**: Cap dinámico según posiciones mínimas
- ✅ **Cálculo correcto de turnover**: Anualización apropiada
- ✅ **Logging optimizado**: Reducido spam en logs

## 📊 Suite de Visualizaciones

### Prioridad 1: Análisis de Performance (9 gráficos)
1. **Cumulative Returns Charts** (3)
   - Portfolio vs S&P 500 para cada estrategia
   - Box de estadísticas integrado
   - Formato profesional con % en ejes

2. **Drawdown Charts** (3)
   - Efecto "underwater" con fill rojo
   - Máximo drawdown y fecha destacados
   - Esencial para gestión de riesgo

3. **Rolling Sharpe Charts** (3)
   - Sharpe ratio móvil de 12 meses
   - Líneas de referencia en 0 y 1.0
   - Análisis de estabilidad de estrategias

### Prioridad 2: Análisis Integral (5+ gráficos)
4. **Portfolio Composition Charts** (3)
   - Pie charts con estadísticas de concentración
   - Filtro automático de pesos menores a 1%
   - Métricas de diversificación

5. **Monthly Returns Heatmaps** (3)
   - Matriz año-mes con código de colores
   - Análisis de estacionalidad
   - Estadísticas de consistencia

6. **Risk-Return Scatter Plot** (1)
   - Todas las estrategias vs benchmark
   - Colormap por Sharpe ratio
   - Identificación de frontera eficiente

7. **Turnover Analysis Charts** (1)
   - Impacto de costos de transacción
   - Comparación gross vs net returns
   - Análisis de eficiencia operacional

## 📈 Métricas de Performance Calculadas

### Métricas Básicas
- **Total Return**: Retorno acumulado del período
- **CAGR**: Tasa de crecimiento anual compuesta
- **Volatilidad Anualizada**: Riesgo del portafolio
- **Sharpe Ratio**: Retorno ajustado por riesgo

### Métricas Avanzadas
- **Information Ratio**: Excess return / Tracking error
- **Tracking Error**: Volatilidad del excess return vs benchmark
- **Beta**: Sensibilidad al mercado
- **Hit Rate**: % períodos ganando al benchmark
- **Maximum Drawdown**: Pérdida máxima desde peak

### Métricas de Transacción
- **Average Turnover**: Promedio de cambios por rebalanceo
- **Annual Turnover**: Turnover anualizado según frecuencia
- **Transaction Cost Impact**: Impacto real en retornos

## 🔄 Flujo de Ejecución

### Paso 1: Obtención de Datos
```python
# Descarga 26 símbolos (25 activos + benchmark)
# Período: 5 años (2020-2025)
# Fuente: Yahoo Finance
# Limpieza: Duplicados, NaNs, validación
```

### Paso 2: Filtros de Calidad
```python
# Cobertura mínima: 80%
# Sesiones mínimas: Adaptativo (80% de disponibles, mín 20)
# Fallback: Relajar criterios si <3 activos pasan
```

### Paso 3: Cálculo de Métricas
```python
# 11 métricas por activo
# Outlier clipping: Percentiles 1-99
# Ranking por Sharpe ratio
```

### Paso 4: Screening Enhanced
```python
# Normalización robusta de métricas
# Score compuesto: 70% Sharpe + 20% Low Vol + 10% Momentum
# Selección top 15 activos
```

### Paso 5: Optimización Actual
```python
# 3 estrategias optimizadas con datos completos
# Pesos respetando restricciones
# Métricas esperadas calculadas
```

### Paso 6: Backtesting Rolling
```python
# 57 períodos de rebalanceo
# Ventana móvil de 2 años
# Rebalanceo mensual
# Costos de transacción aplicados
```

### Paso 7: Generación de Visualizaciones
```python
# 14 gráficos automáticos
# 2 prioridades de análisis
# Estadísticas integradas
```

### Paso 8: Reporte Final
```python
# Comparación de estrategias
# Recomendaciones automáticas
# Resumen de configuración
```

## 📊 Resultados Típicos

### Performance Histórica (2022-2025)
```
Max Sharpe Strategy:
├── Total Return: 126.21% vs S&P 500 76.42%
├── CAGR: 19.10% vs 12.92%
├── Sharpe Ratio: 1.07 vs 0.67
├── Max Drawdown: -20.58%
├── Information Ratio: 0.77
└── Beta: 0.83

Min Variance Strategy:
├── Total Return: 99.70%
├── CAGR: 15.96%
├── Sharpe Ratio: 0.95
├── Max Drawdown: -19.76%
└── Volatilidad: 14.70%

Risk Parity Strategy:
├── Total Return: 100.44%
├── CAGR: 16.05%
├── Sharpe Ratio: 0.99
├── Max Drawdown: -17.82% (mejor)
└── Beta: 0.74 (más conservador)
```

### Análisis de Transacciones
```
Turnover Promedio: ~12-13% por rebalanceo
Turnover Anualizado: ~145-155%
Costo Promedio: ~0.22% anual
Rebalanceos Completados: 57
```

## 🛠️ Instalación y Uso

### Requisitos
```bash
pip install yfinance pandas numpy scipy matplotlib seaborn
```

### Ejecución
```bash
python professional_portfolio_optimizer_fixed.py
```

### Salida Esperada
- Métricas detalladas en consola
- 14 gráficos automáticos
- Recomendaciones de estrategia
- Tiempo de ejecución: ~2 minutos

## 🔬 Supuestos y Limitaciones

### Supuestos del Modelo
- **Mercados eficientes**: Los precios reflejan información disponible
- **Distribución normal**: Retornos siguen distribución aproximadamente normal
- **Estacionariedad**: Relaciones históricas se mantienen
- **Liquidez perfecta**: Ejecución inmediata sin impacto en precio
- **Costos fijos**: 15 bps por transacción constante

### Limitaciones Conocidas
- **Lookback bias**: Optimización conoce activos que "sobreviven"
- **Régimen dependence**: Performance puede cambiar con condiciones macro
- **Small sample**: 5 años pueden no capturar todos los regímenes
- **Transaction costs**: Modelo simplificado, no incluye spread o market impact

### Consideraciones de Implementación
- **Frecuencia**: Mensual puede ser subóptima para algunos factores
- **Universo fijo**: 25 activos puede limitar diversificación
- **Risk model**: Matriz de covarianza histórica vs modelos factoriales

## 🔄 Posibles Mejoras Futuras

### Técnicas
- [ ] Modelos de riesgo factoriales (Fama-French)
- [ ] Optimización Black-Litterman
- [ ] Machine Learning para return forecasting
- [ ] Regime-aware asset allocation

### Datos
- [ ] Incorporar más clases de activos
- [ ] Datos fundamentales y macroeconómicos
- [ ] Análisis de sentiment y flows

### Operacional
- [ ] Live trading integration
- [ ] Portfolio rebalancing alerts
- [ ] Risk monitoring dashboard
- [ ] Performance attribution analysis

## 📊 Ejemplos de Visualizaciones

### 🎯 Prioridad 1: Análisis de Performance

#### Cumulative Returns - Comparación vs S&P 500
Evolución del valor del portafolio vs benchmark a lo largo del tiempo.

**Max Sharpe Strategy:**
![Cumulative Returns Max Sharpe](images/Cumulative%20Returns%20Max%20Sharpe%20Strategy%20vs%20S&P500.png)

**Min Variance Strategy:**
![Cumulative Returns Min Variance](images/Cumulative%20Returns%20Min%20Variance%20Strategy.png)

**Risk Parity Strategy:**
![Cumulative Returns Risk Parity](images/Cumulative%20Returns%20Risk%20Parity%20Strategy.png)

#### Drawdown Analysis - Gestión de Riesgo
Análisis "underwater" mostrando pérdidas máximas desde picos históricos.

**Max Sharpe Strategy:**
![Drawdown Max Sharpe](images/Drawdowns%20Analysis%20Max%20Sharpe%20Strategy.png)

**Min Variance Strategy:**
![Drawdown Min Variance](images/Drawdowns%20Analysis%20Min%20Variance.png)

**Risk Parity Strategy:**
![Drawdown Risk Parity](images/Drawdowns%20Analysis%20Risk%20Parity.png)

#### Rolling Sharpe Ratio - Estabilidad de Performance
Sharpe ratio móvil de 12 meses para evaluar consistencia de la estrategia.

**Max Sharpe Strategy:**
![Rolling Sharpe Max Sharpe](images/12%20month%20Rolling%20Sharpe%20Ratio.png)

**Min Variance Strategy:**
![Rolling Sharpe Min Variance](images/12%20month%20Rolling%20Sharpe%20Ratio%20Min%20Variance%20Strategy.png)

**Risk Parity Strategy:**
![Rolling Sharpe Risk Parity](images/12%20month%20Rolling%20Sharpe%20Ratio%20Risk%20Parity%20Strategy.png)

### 🔍 Prioridad 2: Análisis Integral

#### Portfolio Composition - Distribución de Activos
Composición actual de cada estrategia con métricas de concentración.

**Max Sharpe Strategy:**
![Portfolio Composition Max Sharpe](images/Portfolio%20Composition%20Max%20Sharpe%20Strategy.png)

**Min Variance Strategy:**
![Portfolio Composition Min Variance](images/Portfolio%20Composition%20Min%20Variance%20strategy.png)

**Risk Parity Strategy:**
![Portfolio Composition Risk Parity](images/Portfolio%20Composition%20Risk%20Parity.png)

#### Monthly Returns Heatmap - Análisis de Estacionalidad
Patrones mensuales de retornos para identificar estacionalidad y consistencia.

**Max Sharpe Strategy:**
![Monthly Heatmap Max Sharpe](images/Monthly%20returns%20Heatmap%20Sharpe%20Strategy.png)

**Min Variance Strategy:**
![Monthly Heatmap Min Variance](images/Monthly%20returns%20Heatmap%20Min%20Variance%20Strategy.png)

**Risk Parity Strategy:**
![Monthly Heatmap Risk Parity](images/Monthly%20returns%20Heatmap%20Risk%20Parity%20Strategy.png)

#### Risk-Return Analysis - Frontera Eficiente
Análisis comparativo de todas las estrategias en el espacio riesgo-retorno.

![Risk Return Analysis](images/Risk%20Return%20Analysis.png)

#### Turnover Analysis - Impacto de Costos de Transacción
Análisis del impacto de costos operacionales en el performance neto.

![Turnover Analysis](images/Annual%20Turnover%20and%20Impact%20of%20transactions%20costs.png)

---

### 📈 Interpretación de Resultados

#### Performance Destacada
- **Max Sharpe Strategy**: Mejor ratio riesgo-retorno (126.21% vs 76.42% del S&P 500)
- **Risk Parity Strategy**: Menor drawdown máximo (-17.82%)
- **Min Variance Strategy**: Mayor estabilidad con menor volatilidad

#### Insights Clave
1. **Outperformance Consistente**: Todas las estrategias superan al benchmark
2. **Gestión de Riesgo**: Drawdowns controlados vs mercado general
3. **Eficiencia Operacional**: Costos de transacción manejables (~0.22% anual)
4. **Diversificación**: Portafolios bien balanceados sin concentración excesiva

---

**Disclaimer**: Este código es para fines educativos y de investigación. No constituye asesoría de inversión. Siempre consulte con un profesional financiero antes de tomar decisiones de inversión.
