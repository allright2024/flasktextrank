var boxs = document.querySelectorAll('.box');

total = 0

boxs.forEach(element => {
    console.log(element.getAttribute('value'))
    total += parseFloat(element.getAttribute('value'))
});

boxs.forEach(element => {
    element.style.width = parseFloat(element.getAttribute('value')) / total * 100 + '%';
});