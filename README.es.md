# Hanglish IME

**Habla coreano. Escríbelo con letras latinas.**

[English](README.md) · [한국어](README.ko.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md) · [Español](README.es.md)

Hanglish es un método de entrada experimental para personas que **pueden hablar coreano, pero tienen dificultades para escribirlo en hangul**. Escribe las palabras coreanas que conoces con letras latinas y Hanglish te propondrá cómo escribirlas en hangul. No necesitas aprender la distribución de un teclado coreano para empezar.

Por ejemplo, si sabes decir *jeoneun haksaengimnida* («Soy estudiante»), puedes escribir esa pronunciación en lugar de introducir cada letra coreana por separado.

| Lo que escribes | Propuesta en coreano | Significado |
|---|---|---|
| `jeoneun haksaengimnida` | 저는 학생입니다 | Soy estudiante. |
| `imnida` | 입니다 | Una forma formal y cortés de «ser». |
| `testreul sihaenghanda` | test를 시행한다 | Realizar una prueba; se conserva la palabra inglesa `test`. |

Estos ejemplos forman parte de las pruebas básicas del motor incluido. No garantizan que todas las frases se conviertan correctamente.

## Cómo se usa

1. Selecciona **Hanglish** en el menú de entrada de macOS.
2. Escribe la pronunciación coreana con letras latinas, incluidos los espacios entre palabras.
3. Pulsa **Enter** o **Tab** para solicitar propuestas en coreano.
4. Usa **Tab / Shift+Tab** o **Abajo / Arriba** para cambiar de propuesta. Pulsa **Enter** para confirmar tu elección.
5. Pulsa **Esc** si prefieres conservar el texto original en letras latinas.

La entrada se basa en la romanización revisada del coreano (Revised Romanization). Algunas grafías informales habituales pueden no funcionar. Es una herramienta de escritura: debes introducir palabras coreanas, no una frase en inglés para traducir. Tampoco transcribe la voz.

La pronunciación coreana no determina una única escritura. Hanglish intenta recuperar la palabra que querías escribir, pero puede equivocarse. Elegir entre propuestas requiere reconocer algo de coreano escrito; si no puedes leer hangul en absoluto, quizá necesites ayuda para comprobar el resultado. Aún no se muestran guías de pronunciación ni traducciones de las propuestas.

## Disponibilidad

**Prototipo nativo actual: macOS 0.2.5, Apple Silicon y macOS 14 o posterior.** Aún no hay métodos de entrada implementados para Windows, Android o iOS.

**Todavía no se ha publicado ninguna GitHub Release.** El repositorio es privado por ahora; la licencia del proyecto y las condiciones de distribución pública de los modelos siguen en preparación.

Si has recibido el instalador de desarrollo, consulta la [guía de instalación para macOS (English)](native/macos/README.md) o la [versión en 한국어](native/macos/README.ko.md). El instalador completo incluye el motor, los modelos y el entorno de ejecución de Python. Una vez instalado, convierte el texto localmente y no lo envía a un servidor. No necesitas un servidor de desarrollo, Tailscale ni instalar Python por separado.

El instalador de desarrollo tiene una firma ad-hoc. Aún no se han completado la firma con Developer ID ni la notarización de Apple, y se sigue probando la compatibilidad con distintas aplicaciones.

## Limitaciones

- Es un método de entrada experimental, no un corrector ortográfico o gramatical terminado.
- Los cambios de pronunciación, las palabras ambiguas, el vocabulario poco frecuente y la mezcla de coreano e inglés pueden producir propuestas incorrectas.
- En 1.000 frases de regresión ya utilizadas durante el desarrollo, el motor seleccionado acertó 954 veces con su primera propuesta; en 975 casos había una respuesta aceptada entre las cinco primeras. **No es una medición independiente de precisión en uso real.** Quedaron 46 errores en la primera propuesta.
- Aún no se puede editar dentro de una composición ni volver a convertir palabras individuales.
- La interfaz web experimental es una herramienta local de desarrollo y queda fuera del ámbito de distribución de este repositorio.

## Para desarrolladores

El repositorio contiene el código del conversor, el método de entrada nativo para macOS, herramientas de compilación, pruebas y notas de desarrollo. No incluye los pesos de los modelos, datos descargados o generados, resultados de experimentos locales ni imágenes del instalador.

**Un clon nuevo todavía no permite ejecutar el motor aprendido completo ni compilar el instalador independiente.** Los archivos necesarios se enumeran en el [manifiesto del motor](native/macos/Packaging/engine-files.json); algunos siguen usando rutas de experimentos locales. Antes de distribuirlo públicamente hace falta un proceso reproducible de descarga y verificación de esos recursos.

En un Mac con Xcode Command Line Tools, ejecuta desde la raíz del repositorio las comprobaciones nativas que no requieren modelos:

```sh
bash native/macos/test.sh
```

Comprueban el estado de composición, el protocolo de comunicación con el proceso del motor y los tipos de Swift. No demuestran que funcionen la selección real de la fuente de entrada, la escritura o la conversión del modelo. Algunos archivos de `tests/` son diagnósticos manuales de teclado que se ejecutan al importarlos: **no ejecutes indiscriminadamente toda la suite de pytest**. Hay que separarlos antes de ofrecer un comando general de pruebas.

## Más información

La mayoría de los registros detallados de desarrollo están actualmente en coreano:

- [Instalación y desarrollo en macOS (English)](native/macos/README.md)
- [Estructura del paquete (한국어)](docs/macos-package.md)
- [Preparación de GitHub y Releases (한국어)](docs/github-release.md)
- [Qué funciona y qué necesita mejorar (한국어)](docs/development-retrospective.md)
- [Entrenamiento y procedencia de los datos (한국어)](docs/neural-training.md)
- [Comparación de plataformas (한국어)](docs/ime-platform-comparison.md)

El README principal está en inglés. Las traducciones deben mantener las mismas plataformas compatibles, limitaciones y estado de publicación.
