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

**Prototipo nativo actual: macOS 0.2.7 (compilación 9), Apple Silicon, con macOS 14 o posterior como destino mínimo.** Las pruebas locales se han realizado en macOS 27. Las demás versiones, incluido macOS 14, se ofrecen como **versión preliminar sin verificar**; el destino mínimo no garantiza la compatibilidad. Aún no hay métodos de entrada implementados para Windows, Android o iOS.

**Todavía no se ha publicado ninguna GitHub Release.** El repositorio es privado por ahora. El código original usa condiciones no comerciales; la distribución pública de los modelos sigue en preparación.

Para instalar y configurar la fuente de entrada, consulta la [guía de instalación para macOS (English)](native/macos/README.md) o la [versión en 한국어](native/macos/README.ko.md). El instalador completo incluye el motor, los modelos y el entorno de ejecución de Python. Una vez instalado, convierte el texto localmente y no lo envía a un servidor. No necesitas un servidor de desarrollo, Tailscale ni instalar Python por separado.

El instalador usa inglés de forma predeterminada e incluye localización al coreano. El instalador de desarrollo tiene una firma ad-hoc. Los paquetes del modo release requieren firma con Developer ID y notarización de Apple; consulte el manifiesto y el registro de notarización del paquete concreto. Se sigue probando la compatibilidad con distintas aplicaciones.

## Limitaciones

- Es un método de entrada experimental, no un corrector ortográfico o gramatical terminado.
- Los cambios de pronunciación, las palabras ambiguas, el vocabulario poco frecuente y la mezcla de coreano e inglés pueden producir propuestas incorrectas.
- En 1.000 frases de regresión ya utilizadas durante el desarrollo, el motor seleccionado acertó 988 veces con su primera propuesta; en 1.000 casos había una respuesta aceptada entre las cinco primeras. **No es una medición independiente de precisión en uso real.** Quedaron 12 errores en la primera propuesta.
- Aún no se puede editar dentro de una composición ni volver a convertir palabras individuales.
- La interfaz web experimental es una herramienta local de desarrollo y queda fuera del ámbito de distribución de este repositorio.

## Para desarrolladores

El repositorio contiene el código del conversor, el método de entrada nativo para macOS, herramientas de compilación, pruebas. No incluye los pesos de los modelos, datos descargados o generados, resultados de experimentos locales ni imágenes del instalador.

**Un clon nuevo permite ejecutar la conversión básica sin modelos. El motor aprendido completo y el instalador requieren recursos adicionales.** La [guía de desarrollo (en inglés)](native/macos/README.md#build-and-verify) explica las dependencias fijadas, la verificación de recursos y la importación desde una instalación local de confianza. Solo se puede importar un paquete que coincida exactamente con el [archivo de recursos fijados](native/macos/Packaging/runtime-assets.lock.json); después de normalizar los metadatos de procedencia, una instalación antigua de la versión 0.2.5 ya no coincide con el bloqueo actual. Todavía no hay descargas públicas de modelos.

En un Mac con Xcode Command Line Tools, ejecuta desde la raíz del repositorio las comprobaciones nativas que no requieren modelos:

```sh
bash native/macos/test.sh
```

Comprueban el estado de composición, el protocolo con el proceso del motor y los tipos de Swift. No demuestran la selección real de la fuente de entrada, la escritura o la conversión del modelo. Los diagnósticos manuales se han movido a `scripts/manual/` y requieren `--run`; importarlos no envía pulsaciones. Usa las pruebas básicas de la [guía de desarrollo](native/macos/README.md#build-and-verify). Las pruebas de investigación pueden necesitar recursos adicionales.

## Licencia

El código original y los diccionarios propios usan [CC BY-NC 4.0](LICENSE). Permite compartir y adaptar para fines no comerciales con atribución; al compartir material modificado, deben indicarse los cambios. Es software con código disponible bajo condiciones no comerciales, no código abierto según la definición de la OSI. Consulta el [alcance de la licencia (en inglés)](LICENSE) y los [avisos de terceros (en inglés)](THIRD_PARTY_NOTICES.md). Los diccionarios y modelos externos conservan sus condiciones originales.
