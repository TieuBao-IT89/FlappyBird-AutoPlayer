"use strict";
(() => {
  const canvas = document.querySelector('#game'), ctx = canvas.getContext('2d');
  const W = 360, H = 600, floor = 528, gravity = 740, jump = -260, speed = 100;
  const bird = { x: 105, y: 270, vy: 0, r: 12 };
  let pipes, state = 'ready', score = 0, lastTime = 0;
  const newPipe = (x, n) => ({ x, opening: 255 + Math.sin(n * 1.7) * 85, passed: false });
  function reset() { bird.y=270; bird.vy=0; pipes=[newPipe(370,0),newPipe(580,1),newPipe(790,2)]; score=0; state='ready'; }
  reset();
  function input() {
    if (state === 'dead') reset();
    if (state === 'ready') state='playing';
    bird.vy=jump;
  }
  window.addEventListener('keydown', e => { if(e.code==='Space' || e.code==='ArrowUp'){ e.preventDefault(); input(); } });
  canvas.addEventListener('pointerdown', input);
  function update(dt) {
    if(state!=='playing') return;
    bird.vy += gravity*dt;
    bird.y += bird.vy*dt;
    for(const p of pipes) {
      p.x -= speed*dt;
      if(p.x+52 < bird.x && !p.passed) { score++; p.passed=true; }
      if(bird.x+bird.r>p.x && bird.x-bird.r<p.x+52 && (bird.y-bird.r<p.opening-84 || bird.y+bird.r>p.opening+84)) state='dead';
    }
    if(pipes[0].x < -60) {pipes.shift();pipes.push(newPipe(pipes[pipes.length-1].x+210,score+3));}
    if(bird.y+bird.r>floor || bird.y-bird.r<0) state='dead';
  }
  function draw() {
    ctx.fillStyle='#80cdf1';ctx.fillRect(0,0,W,H);
    ctx.fillStyle='#e0f6fc';
    for(let i=0;i<3;i++) {ctx.beginPath();ctx.ellipse(45+i*140,85+(i%2)*65,38,12,0,0,Math.PI*2);ctx.fill();}
    // Pipes: vivid green, separated by 168 px openings.
    for(const p of pipes) {
      ctx.fillStyle='#37b83e';ctx.fillRect(p.x,0,52,p.opening-84);ctx.fillRect(p.x,p.opening+84,52,floor-p.opening-84);
      ctx.fillStyle='#48dc4b';ctx.fillRect(p.x-3,p.opening-102,58,18);ctx.fillRect(p.x-3,p.opening+84,58,18);
      ctx.fillStyle='#178b27';ctx.fillRect(p.x+45,0,7,p.opening-102);ctx.fillRect(p.x+45,p.opening+102,7,floor-p.opening-102);
    }
    ctx.fillStyle='#c3a45d';ctx.fillRect(0,floor,W,H-floor);ctx.fillStyle='#87c648';ctx.fillRect(0,floor,W,14);
    // Yellow bird sprite is distinct from the green pipe sprites.
    ctx.fillStyle='#fbd030';ctx.beginPath();ctx.arc(bird.x,bird.y,bird.r,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#f89d2b';ctx.beginPath();ctx.ellipse(bird.x-5,bird.y+3,7,4,.2,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='white';ctx.beginPath();ctx.arc(bird.x+5,bird.y-4,4,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#2e3438';ctx.beginPath();ctx.arc(bird.x+6,bird.y-4,2,0,Math.PI*2);ctx.fill();
    ctx.textAlign='center';ctx.fillStyle='white';ctx.strokeStyle='#213b4c';ctx.lineWidth=4;ctx.font='bold 34px sans-serif';ctx.strokeText(String(score),W/2,55);ctx.fillText(String(score),W/2,55);
    if(state!=='playing') {ctx.font='bold 22px sans-serif';ctx.strokeText(state==='ready'?'SPACE / CLICK TO START':'GAME OVER - SPACE TO RESTART',W/2,200);ctx.fillText(state==='ready'?'SPACE / CLICK TO START':'GAME OVER - SPACE TO RESTART',W/2,200);}
  }
  function loop(ms) {const dt=Math.min((ms-lastTime)/1000||0,.035);lastTime=ms;update(dt);draw();requestAnimationFrame(loop);}
  requestAnimationFrame(loop);
})();
