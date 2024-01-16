<h1> 
    ✒️ Signess
</h1>

<h3>
    Signess - библиотека для определения определения владельца подписи на основе нейронной сети от Fedot
</h3>


</br>



<h2>
    🛠️ Инструменты:
</h2>

- Python
- Fedot
- Inskrib



</br>



<h2>
  🚀 Установка:
</h2>
    
- `pip install signess`

<h3>
    Запускаем, не работет, ура! 🗿🚬
</h3>



</br>



<h2>
 📺 Демо:
</h2>

- <a href="https://colab.research.google.com/drive/1y3O0GpI3eiRyukHsi1wMb7GeCacmVfMA">Google Colab</a>
- Пример использования - <a href="https://github.com/ElishaFlacon/signess/tree/main/example">example</a>
- Пример использования в реальном проекте - <a href="https://github.com/ElishaFlacon/signess-app">example</a>


</br>



<h2>
⚡ Немного дополнительной информации:
</h2>


- При сборке датасета из документов есть большая вероятность появления ошибок, особенно если документы подготовленны без учета правил использования библиотеки Inskrib:
    - Inskrib не может автоматически переворачивать документы, но для этого есть отдельный метод в Autographs
    - Inskrib плохо работает с документами, в которых есть несколько подписей в разных частях документа, или если есть текст такого же цвета, что и подпись
    - Если печать имеет много разрывов, то метод удаления печати может не сработать
- Метод для загрузки модели `.load` работает не корректно до версии FEDOT >= 0.7.3
- P.S. Все баги и недочеты - это фичи




<br/>
<br/>
<br/>
<br/>
<br/>
<br/>



<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=d179b8&height=64&section=footer"/>
</p>
