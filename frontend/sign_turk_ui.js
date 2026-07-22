const { useState, useEffect, useRef, useCallback, useMemo, useLayoutEffect } = React;

/* ============================================================
   CUSTOM CURSOR — premium ring + dot, magnetic on [data-magnet]
   ============================================================ */
function CustomCursor(){
  const ringRef = useRef(null);
  const dotRef  = useRef(null);
  useEffect(() => {
    const isTouch = matchMedia('(pointer:coarse)').matches;
    if (isTouch) return;
    document.body.classList.add('has-cursor');

    const target = { x: window.innerWidth/2, y: window.innerHeight/2 };
    const ring   = { x: target.x, y: target.y };
    const dot    = { x: target.x, y: target.y };
    let raf = null;

    const onMove = (e) => { target.x = e.clientX; target.y = e.clientY; };
    const onDown = () => ringRef.current && ringRef.current.classList.add('click');
    const onUp   = () => ringRef.current && ringRef.current.classList.remove('click');

    const enterables = 'a, button, [role="button"], [data-magnet], input, select, textarea, .nav-list button, .topk-row, .list-row, .feature-card, .tag, .pipeline-step';
    const onOver = (e) => {
      if (e.target.closest(enterables)) ringRef.current?.classList.add('hover');
    };
    const onOut  = (e) => {
      if (e.target.closest(enterables)) ringRef.current?.classList.remove('hover');
    };

    const tick = () => {
      ring.x += (target.x - ring.x) * 0.16;
      ring.y += (target.y - ring.y) * 0.16;
      dot.x  += (target.x - dot.x)  * 0.45;
      dot.y  += (target.y - dot.y)  * 0.45;
      if (ringRef.current) ringRef.current.style.transform = `translate(${ring.x - 15}px, ${ring.y - 15}px)`;
      if (dotRef.current)  dotRef.current.style.transform  = `translate(${dot.x - 2.5}px, ${dot.y - 2.5}px)`;
      raf = requestAnimationFrame(tick);
    };
    tick();

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mousedown', onDown);
    window.addEventListener('mouseup',   onUp);
    document.addEventListener('mouseover', onOver);
    document.addEventListener('mouseout',  onOut);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('mouseup',   onUp);
      document.removeEventListener('mouseover', onOver);
      document.removeEventListener('mouseout',  onOut);
      document.body.classList.remove('has-cursor');
    };
  }, []);
  return (
    <>
      <div ref={ringRef} className="cursor-ring" />
      <div ref={dotRef}  className="cursor-dot" />
    </>
  );
}

/* ============================================================
   HASH ROUTING + STORED USER (refresh persistence)
   ============================================================ */
function useHashRoute(){
  const read = () => (window.location.hash || '').replace(/^#\/?/, '') || 'home';
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const navigate = useCallback((to) => {
    const target = '#/' + to.replace(/^\//,'');
    if (window.location.hash !== target) window.location.hash = target;
  }, []);
  return [route, navigate];
}

function useStoredUser(){
  const [user, setUserState] = useState(() => {
    try { return JSON.parse(localStorage.getItem('tsl_user') || 'null'); } catch { return null; }
  });
  const setUser = useCallback((u) => {
    if (u) localStorage.setItem('tsl_user', JSON.stringify(u));
    else   localStorage.removeItem('tsl_user');
    setUserState(u);
  }, []);
  return [user, setUser];
}

async function authFetch(input, init={}){
  const options = {...init};
  const headers = new Headers(init.headers || {});
  try{
    const user = JSON.parse(localStorage.getItem('tsl_user') || 'null');
    if (user?.access_token) headers.set('Authorization', `Bearer ${user.access_token}`);
  } catch {}
  options.headers = headers;
  const response = await fetch(input, options);
  if (response.status === 401){
    localStorage.removeItem('tsl_user');
  }
  return response;
}

/* ============================================================
   GLOBAL MOUSE PARALLAX (subtle background shift)
   ============================================================ */
function useMouseParallax(){
  useEffect(() => {
    if (matchMedia('(pointer:coarse)').matches) return;
    let raf = null;
    const onMove = (e) => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        const x = (e.clientX / window.innerWidth  - 0.5) * 2;
        const y = (e.clientY / window.innerHeight - 0.5) * 2;
        document.documentElement.style.setProperty('--mx', x.toFixed(3));
        document.documentElement.style.setProperty('--my', y.toFixed(3));
        raf = null;
      });
    };
    window.addEventListener('mousemove', onMove, { passive:true });
    return () => window.removeEventListener('mousemove', onMove);
  }, []);
}

/* ============================================================
   ELEGANT QUANTUM WAVES — flowing neural ribbons (canvas)
   ============================================================ */
function QuantumWaves({ density=4, mouse=true }){
  const cvRef = useRef(null);
  useEffect(() => {
    const canvas = cvRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf = null, t0 = performance.now();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const mx = { x: 0.5, y: 0.5 };

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      canvas.width  = r.width  * dpr;
      canvas.height = r.height * dpr;
      ctx.setTransform(dpr,0,0,dpr,0,0);
    };
    resize();

    const onMove = (e) => {
      const r = canvas.getBoundingClientRect();
      mx.x = (e.clientX - r.left) / r.width;
      mx.y = (e.clientY - r.top)  / r.height;
    };
    if (mouse) window.addEventListener('mousemove', onMove);
    window.addEventListener('resize', resize);

    const palette = [
      'rgba(201,178,137,', // champagne
      'rgba(123,149,179,', // blue
      'rgba(168,142,94,',  // gold-deep
      'rgba(216,195,158,', // champagne-2
    ];

    const draw = (now) => {
      const t = (now - t0) / 1000;
      const w = canvas.width  / dpr;
      const h = canvas.height / dpr;
      ctx.clearRect(0,0,w,h);

      const ribbons = density;
      for (let r = 0; r < ribbons; r++){
        const phase = r * 0.7 + t * 0.45;
        const amp   = h * 0.12 * (1 + (mouse ? (mx.y - 0.5) * 0.4 : 0));
        const baseY = h * (0.25 + r * 0.18);
        const colorBase = palette[r % palette.length];
        const alpha = 0.10 + (r % 2) * 0.04;

        ctx.beginPath();
        ctx.moveTo(0, baseY);
        for (let x = 0; x <= w; x += 8){
          const px = x / w;
          const y = baseY
            + Math.sin(px * 6 + phase)        * amp
            + Math.sin(px * 11 + phase * 1.6) * amp * 0.4
            + (mouse ? (mx.x - 0.5) * 30 * Math.sin(px * 4 + phase) : 0);
          ctx.lineTo(x, y);
        }
        const grad = ctx.createLinearGradient(0,0,w,0);
        grad.addColorStop(0,    colorBase + '0)');
        grad.addColorStop(0.25, colorBase + alpha + ')');
        grad.addColorStop(0.75, colorBase + alpha + ')');
        grad.addColorStop(1,    colorBase + '0)');
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.1;
        ctx.stroke();

        // Twin ribbon offset
        ctx.beginPath();
        ctx.moveTo(0, baseY + 8);
        for (let x = 0; x <= w; x += 8){
          const px = x / w;
          const y = baseY + 8
            + Math.sin(px * 6 + phase + 0.6) * amp * 0.85
            + Math.sin(px * 11 + phase * 1.6 + 0.6) * amp * 0.36;
          ctx.lineTo(x, y);
        }
        ctx.strokeStyle = grad;
        ctx.lineWidth = 0.7;
        ctx.globalAlpha = 0.8;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      if (mouse) window.removeEventListener('mousemove', onMove);
    };
  }, [density, mouse]);
  return <canvas ref={cvRef} className="waves" />;
}

/* ============================================================
   PARALLAX ORBS — mouse-reactive
   ============================================================ */
function ParallaxOrbs({ children }){
  const ref = useRef(null);
  useEffect(() => {
    if (matchMedia('(pointer:coarse)').matches) return;
    let raf = null;
    const onMove = (e) => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        const orbs = ref.current?.querySelectorAll('.orb, .auth-orb');
        if (orbs){
          orbs.forEach((o, i) => {
            const factor = (i + 1) * 6;
            const x = (e.clientX / window.innerWidth  - 0.5) * factor;
            const y = (e.clientY / window.innerHeight - 0.5) * factor;
            o.style.transform = `translate(${x}px, ${y}px)`;
          });
        }
        raf = null;
      });
    };
    window.addEventListener('mousemove', onMove, { passive:true });
    return () => window.removeEventListener('mousemove', onMove);
  }, []);
  return <div ref={ref}>{children}</div>;
}

/* ============================================================
   REVEAL ON SCROLL
   ============================================================ */
function Reveal({ children, delay=0, className='' }){
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting){ setTimeout(() => el.classList.add('in'), delay); io.unobserve(el); }
      });
    }, { threshold: 0.12 });
    io.observe(el);
    return () => io.disconnect();
  }, [delay]);
  return <div ref={ref} className={`reveal ${className}`}>{children}</div>;
}

/* ============================================================
   MAGNETIC HOVER — attaches gentle pull to [data-magnet]
   ============================================================ */
function useMagnets(){
  useEffect(() => {
    if (matchMedia('(pointer:coarse)').matches) return;
    const els = Array.from(document.querySelectorAll('[data-magnet]'));
    const handlers = [];
    els.forEach(el => {
      let raf = null;
      const onMove = (e) => {
        if (raf) return;
        raf = requestAnimationFrame(() => {
          const r = el.getBoundingClientRect();
          const dx = e.clientX - (r.left + r.width / 2);
          const dy = e.clientY - (r.top  + r.height / 2);
          el.style.transform = `translate(${dx * 0.16}px, ${dy * 0.22}px)`;
          raf = null;
        });
      };
      const onLeave = () => { el.style.transform = ''; };
      el.addEventListener('mousemove', onMove);
      el.addEventListener('mouseleave', onLeave);
      handlers.push([el, onMove, onLeave]);
    });
    return () => handlers.forEach(([el, m, l]) => {
      el.removeEventListener('mousemove', m);
      el.removeEventListener('mouseleave', l);
    });
  });
}

/* ============================================================
   SPLASH
   ============================================================ */
function Splash(){
  return (
    <div className="splash">
      <div className="splash-grid" />
      <div className="splash-inner">
        <div className="splash-mark">Sign<em>Turk</em></div>
        <div className="splash-sub">Sign Language Intelligence</div>
        <div className="splash-bar" />
      </div>
    </div>
  );
}

/* ============================================================
   LANDING PAGE
   ============================================================ */
function LandingPage({ onLaunch }){
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    document.body.classList.add('scrollable');
    const onScroll = () => setScrolled(window.scrollY > 18);
    window.addEventListener('scroll', onScroll);
    return () => { document.body.classList.remove('scrollable'); window.removeEventListener('scroll', onScroll); };
  }, []);
  useMagnets();

  return (
    <div className="landing page-enter">
      <div className="ambient-bg" />
      <div className="ambient-grid" />
      <ParallaxOrbs>
        <div className="orb l1" />
        <div className="orb l2" />
        <div className="orb l3" />
      </ParallaxOrbs>
      <QuantumWaves density={4} mouse={true} />

      <div className="landing-inner">
        <nav className={`nav ${scrolled ? 'scrolled' : ''}`}>
          <div className="nav-brand" onClick={() => window.scrollTo({top:0, behavior:'smooth'})}>
            <div className="brand-mark"><span>S</span></div>
            <span>SignTurk</span>
          </div>
          <div className="nav-links">
            <a className="nav-link" onClick={() => document.getElementById('features').scrollIntoView({behavior:'smooth'})}>Features</a>
            <a className="nav-link" onClick={() => document.getElementById('how').scrollIntoView({behavior:'smooth'})}>Pipeline</a>
            <a className="nav-link" onClick={() => document.getElementById('tech').scrollIntoView({behavior:'smooth'})}>Technology</a>
            <a className="nav-link" onClick={() => document.getElementById('impact').scrollIntoView({behavior:'smooth'})}>Impact</a>
          </div>
          <button className="nav-cta" data-magnet onClick={onLaunch}>Launch Platform <span>→</span></button>
        </nav>

        <section className="hero">
          <div className="fade-up">
            <div className="hero-tag"><i/>Türk İşaret Dili Intelligence</div>
          </div>
          <h1 className="hero-title fade-up d1">
            Sign language,<br/>
            <em>understood</em> instantly.
          </h1>
          <p className="hero-sub fade-up d2">
            A real-time Turkish Sign Language recognition platform powered by deep learning.
            Camera-based gesture inference across 179 signs, built for accessible real-time interaction.
          </p>
          <div className="hero-ctas fade-up d3">
            <button className="btn-primary" data-magnet onClick={onLaunch}>Launch Platform<span>→</span></button>
            <a className="btn-ghost" onClick={() => document.getElementById('how').scrollIntoView({behavior:'smooth'})}>See how it works</a>
          </div>

          <Reveal className="hero-visual">
            <div className="hero-card">
              <div className="hero-card-inner">
                <div className="hero-grid-bg" />
                <div className="hero-demo">
                  <div className="hero-demo-left">
                    <div className="hero-hand">
                      <div className="hero-ring" />
                      <svg viewBox="0 0 120 160" style={{width:'100%',height:'100%',position:'relative',zIndex:1}}>
                        <defs>
                          <linearGradient id="handG" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0%" stopColor="#c9b289"/>
                            <stop offset="100%" stopColor="#7b95b3"/>
                          </linearGradient>
                        </defs>
                        <path d="M60 145 C 40 140 30 120 30 95 L 30 75 C 30 70 34 66 38 66 C 42 66 46 70 46 75 L 46 85 L 46 55 C 46 50 50 46 54 46 C 58 46 62 50 62 55 L 62 85 L 62 40 C 62 35 66 31 70 31 C 74 31 78 35 78 40 L 78 85 L 78 50 C 78 45 82 41 86 41 C 90 41 94 45 94 50 L 94 95 C 94 120 82 140 72 145 Z" fill="none" stroke="url(#handG)" strokeWidth="1.6" strokeLinejoin="round"/>
                        {[[38,75],[46,55],[62,40],[78,50],[30,95]].map(([x,y],i) => <circle key={i} cx={x} cy={y} r="3" fill="#c9b289"/>)}
                        {[[60,110],[45,100],[75,105]].map(([x,y],i) => <circle key={i} cx={x} cy={y} r="2" fill="#7b95b3" opacity=".7"/>)}
                      </svg>
                    </div>
                  </div>
                  <div className="hero-demo-right">
                    <div>
                      <div className="demo-label">Detected Sign</div>
                      <div className="demo-result">Merhaba<em>.</em></div>
                      <div className="demo-meta">
                        <span className="mono">class_id: 042</span> · <span className="mono">EN: hello</span>
                      </div>
                    </div>
                    <div>
                      <div className="demo-label">Neural Activity</div>
                      <div className="demo-bars">{Array.from({length:9}).map((_,i) => <i key={i}/>)}</div>
                    </div>
                    <div>
                      <div className="demo-label">Top Predictions</div>
                      <div className="demo-topk">
                        <div className="demo-topk-row"><span>Merhaba · hello</span><span className="mono">94.2%</span></div>
                        <div className="demo-topk-row"><span>Selam · hi</span><span className="mono">3.1%</span></div>
                        <div className="demo-topk-row"><span>Günaydın · morning</span><span className="mono">1.4%</span></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Reveal>
        </section>

        <section className="section" id="features">
          <Reveal className="section-head">
            <div className="section-eyebrow">Capabilities</div>
            <h2 className="section-title">Built for <em>real</em> conversation.</h2>
            <p className="section-sub">A complete platform for live sign language recognition — from camera to understanding, in one continuous loop.</p>
          </Reveal>

          <div className="features-grid">
            {[
              {mark:'L', t:'Live Recognition',  d:'WebSocket-streamed camera frames feed a BiLSTM with Temporal Attention over 16-frame windows.'},
              {mark:'D', t:'179 Classes',       d:'Full Türk İşaret Dili vocabulary covering greetings, everyday words, numbers, and essential communication phrases.'},
              {mark:'A', t:'Avatar Mapping',    d:'Text-to-animation pipeline converts detected signs back to visual playback for two-way conversation.'},
              {mark:'M', t:'MediaPipe Tracking',d:'21 landmarks per hand across both hands — a 126-dim vector with relative normalization and joint angles.'},
              {mark:'S', t:'Session History',   d:'User-approved predictions are persisted with confidence and session context.'},
              {mark:'I', t:'Admin Intelligence',d:'User management, activity logs, model telemetry, and inference monitoring — all in one refined control surface.'},
            ].map((f, i) => (
              <Reveal key={i} delay={i*60}>
                <div className="feature-card">
                  <div className="feature-icon">{f.mark}</div>
                  <h3 className="feature-title">{f.t}</h3>
                  <p className="feature-desc">{f.d}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </section>

        <section className="section" id="how">
          <Reveal className="section-head">
            <div className="section-eyebrow">The Pipeline</div>
            <h2 className="section-title">Three steps. <em>One</em> fluid stream.</h2>
            <p className="section-sub">From raw camera frames to recognized words — a continuous inference pipeline designed for real-time interaction.</p>
          </Reveal>
          <div className="steps-grid">
            {[
              {n:'I',   t:'Capture',     d:'Webcam streams 250ms frames. MediaPipe extracts 21 landmarks per hand.'},
              {n:'II',  t:'Understand',  d:'Sequences of 16 frames flow through a BiLSTM with temporal attention to classify signs.'},
              {n:'III', t:'Respond',     d:'Top predictions return with confidence scores. Results map to avatar animations and text output.'},
            ].map((s, i) => (
              <Reveal key={i} delay={i*100}>
                <div className="step">
                  <div className="step-num">{s.n}</div>
                  <h4 className="step-title">{s.t}</h4>
                  <p className="step-desc">{s.d}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </section>

        <section className="section">
          <Reveal>
            <div className="showcase">
              <div className="showcase-inner">
                <div className="showcase-left">
                  <div className="section-eyebrow" style={{marginBottom:18}}>Live Showcase</div>
                  <h3>From webcam to <em>word</em>, in milliseconds.</h3>
                  <p>Open your camera, start signing, and watch recognition stream in real time through the integrated application pipeline.</p>
                  <button className="btn-primary" data-magnet onClick={onLaunch}>Try Live Translation<span>→</span></button>
                  <div className="showcase-stats">
                    <div className="showcase-stat"><span className="serif">179</span><span>Sign classes</span></div>
                    <div className="showcase-stat"><span className="serif">&lt;100ms</span><span>Inference time</span></div>
                    <div className="showcase-stat"><span className="serif">16</span><span>Frame window</span></div>
                  </div>
                </div>
                <div className="showcase-panel">
                  <div className="demo-label">Live Inference</div>
                  <div className="demo-result" style={{fontSize:'clamp(34px, 4vw, 48px)', marginTop:8}}>Teşekkür <em>ederim</em></div>
                  <div className="demo-meta">EN: thank you · <span className="mono">class_id: 107</span></div>
                  <div style={{marginTop:18}}>
                    <div style={{display:'flex',justifyContent:'space-between',marginBottom:6}}>
                      <span className="label">Confidence</span>
                      <span className="mono" style={{fontSize:11,color:'var(--ink-3)'}}>92.8%</span>
                    </div>
                    <div className="progress"><i style={{width:'92.8%'}}/></div>
                  </div>
                  <div className="demo-bars" style={{marginTop:18}}>{Array.from({length:12}).map((_,i) => <i key={i}/>)}</div>
                </div>
              </div>
            </div>
          </Reveal>
        </section>

        <section className="section" id="tech">
          <Reveal className="section-head">
            <div className="section-eyebrow">Under the Hood</div>
            <h2 className="section-title">The AI <em>pipeline</em>, end to end.</h2>
            <p className="section-sub">Every frame passes through six transforms. Each one engineered for accuracy, speed, and linguistic fidelity to Türk İşaret Dili.</p>
          </Reveal>
          <div className="tech-pipeline">
            {[
              {t:'Capture',    d:'JPEG frame',           n:'01'},
              {t:'Landmarks',  d:'MediaPipe · 2 hands',  n:'02'},
              {t:'Normalize',  d:'Wrist-relative · scale', n:'03'},
              {t:'Angles',     d:'30 joint features',    n:'04'},
              {t:'Z-Score',    d:'Training stats',       n:'05'},
              {t:'Predict',    d:'BiLSTM · 179 classes', n:'06'},
            ].map((s, i) => (
              <Reveal key={i} delay={i*50}>
                <div className="pipeline-step">
                  <span className="mono">{s.n}</span>
                  <strong>{s.t}</strong>
                  <em>{s.d}</em>
                </div>
              </Reveal>
            ))}
          </div>
        </section>

        <section className="section" id="impact">
          <Reveal className="section-head">
            <div className="section-eyebrow">Why this matters</div>
            <h2 className="section-title">Accessibility, <em>engineered</em>.</h2>
            <p className="section-sub">SignTurk explores how ordinary webcams and real-time software can support more accessible Turkish Sign Language interaction.</p>
          </Reveal>
          <div className="impact-grid">
            <Reveal>
              <div className="impact-card">
                <h4>Inclusion, built into the product.</h4>
                <p>The prototype demonstrates camera-based recognition for everyday isolated signs while keeping user approval in the translation loop.</p>
              </div>
            </Reveal>
            <Reveal delay={100}>
              <div className="impact-card">
                <h4>A foundation for research.</h4>
                <p>Approved predictions can be stored with confidence and session context in the configured database for later evaluation and model improvement.</p>
              </div>
            </Reveal>
          </div>
        </section>

        <section className="section">
          <Reveal className="section-head">
            <div className="section-eyebrow">Trust</div>
            <h2 className="section-title">Built on <em>measurable</em> ground.</h2>
          </Reveal>
          <Reveal>
            <div className="trust-metrics">
              <div className="trust-metric"><strong>179<em>.</em></strong><span>Sign Classes</span></div>
              <div className="trust-metric"><strong>3.1<em>M</em></strong><span>Model Parameters</span></div>
              <div className="trust-metric"><strong>16<em>×</em>156</strong><span>Input Tensor</span></div>
              <div className="trust-metric"><strong>226<em>.</em></strong><span>Avatar Sequences</span></div>
            </div>
          </Reveal>
        </section>

        <section className="cta-band">
          <Reveal>
            <h2 className="cta-title">Start signing with<br/><em>SignTurk</em>.</h2>
            <p className="cta-sub">An open academic prototype for exploration, evaluation, and continued research.</p>
            <div className="hero-ctas" style={{marginTop:32}}>
              <button className="btn-primary" data-magnet onClick={onLaunch}>Launch Platform<span>→</span></button>
              <a className="btn-ghost" onClick={() => document.getElementById('features').scrollIntoView({behavior:'smooth'})}>Explore capabilities</a>
            </div>
          </Reveal>
        </section>

        <footer className="footer">
          <div className="footer-main">
            <div className="footer-brand">
              <div className="nav-brand"><div className="brand-mark"><span>S</span></div><span>SignTurk</span></div>
              <p>Real-time Turkish Sign Language recognition, powered by deep learning. Built for accessibility, education, and research.</p>
            </div>
            <div className="footer-col">
              <h5>Product</h5>
              <a onClick={onLaunch}>Launch</a>
              <a onClick={() => document.getElementById('features').scrollIntoView({behavior:'smooth'})}>Features</a>
              <a onClick={() => document.getElementById('how').scrollIntoView({behavior:'smooth'})}>Pipeline</a>
              <a onClick={() => document.getElementById('tech').scrollIntoView({behavior:'smooth'})}>Technology</a>
            </div>
            <div className="footer-col">
              <h5>Resources</h5>
              <a onClick={() => document.getElementById('impact').scrollIntoView({behavior:'smooth'})}>Accessibility</a>
              <a>Documentation</a>
              <a>Model Card</a>
              <a>Research</a>
            </div>
            <div className="footer-col">
              <h5>Platform</h5>
              <a>Status</a>
              <a>API</a>
              <a>Terms</a>
              <a>Privacy</a>
            </div>
          </div>
          <div className="footer-bottom">
            <span>© 2026 SignTurk · Graduation project</span>
            <span className="mono" style={{fontSize:10,letterSpacing:'.18em'}}>v2.2 · TÜRK İŞARET DİLİ</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

/* ============================================================
   AUTH SCREEN — split-screen with quantum waves
   ============================================================ */
function AuthScreen({ onAuth, onBack }){
  const [tab, setTab] = useState('login');
  const [form, setForm] = useState({ full_name:'', email:'', password:'', confirm:'' });
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const [ok,  setOk]  = useState('');
  const set = (k) => (e) => { setForm(f => ({ ...f, [k]: e.target.value })); setErr(''); setOk(''); };
  useMagnets();

  const login = async () => {
    if (!form.email || !form.password) return setErr('Please fill in all fields.');
    setLoading(true); setErr('');
    try{
      const res = await authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.login, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ email: form.email, password: form.password })
      });
      const data = await res.json();
      if (!res.ok) return setErr(data.error || 'Sign in failed.');
      onAuth({...data.user, access_token:data.access_token});
    } catch {
      setErr('Cannot reach server. Please ensure the backend is running.');
    } finally { setLoading(false); }
  };

  const register = async () => {
    if (!form.full_name || !form.email || !form.password) return setErr('All fields are required.');
    if (form.password !== form.confirm) return setErr('Passwords do not match.');
    if (form.password.length < 8) return setErr('Password must be at least 8 characters.');
    setLoading(true); setErr(''); setOk('');
    try{
      const res = await authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.register, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ full_name: form.full_name, email: form.email, password: form.password })
      });
      const data = await res.json();
      if (!res.ok) return setErr(data.error || 'Registration failed.');
      setOk('Account created. You may now sign in.');
      setForm(f => ({ ...f, password:'', confirm:'', full_name:'' }));
      setTimeout(() => setTab('login'), 1400);
    } catch {
      setErr('Cannot reach server.');
    } finally { setLoading(false); }
  };

  return (
    <div className="auth page-enter">
      <div className="auth-left">
        <QuantumWaves density={5} mouse={true} />
        <div className="auth-grid-bg" />
        <ParallaxOrbs>
          <div className="auth-orb a1" />
          <div className="auth-orb a2" />
        </ParallaxOrbs>

        <div className="auth-brand fade-in">
          <div className="brand-mark"><span>S</span></div>
          <div className="auth-brand-text">
            <h1>SignTurk</h1>
            <p>Sign Language AI</p>
          </div>
        </div>

        <div className="auth-hero">
          <div className="auth-hero-tag fade-in"><i/>Türk İşaret Dili Intelligence</div>
          <h2 className="fade-in d1">A <em>quieter</em><br/>way to be <em>heard</em>.</h2>
          <p className="fade-in d2">A luxury-grade platform for Turkish Sign Language — real-time recognition, 179 signs, designed for presence and precision.</p>
        </div>

        <div className="auth-features">
          {[
            {m:'L', t:'Live Recognition', d:'Real-time camera inference with BiLSTM attention.'},
            {m:'A', t:'Avatar Mapping',   d:'Two-way communication via animation playback.'},
            {m:'S', t:'Account Access',  d:'Hashed passwords, bearer sessions, persisted history.'},
          ].map((f,i) => (
            <div key={f.m} className={`auth-feature fade-in d${i+3}`}>
              <div className="auth-feature-mark">{f.m}</div>
              <div className="auth-feature-text">
                <strong>{f.t}</strong>
                <span>{f.d}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="auth-right">
        {onBack && <div className="auth-back" onClick={onBack}>← Back to home</div>}
        <div className="auth-card fade-in">
          <div className="auth-tabs">
            <div className={`auth-tab-indicator ${tab==='register' ? 'right' : ''}`} />
            <button className={`auth-tab ${tab==='login'?'active':''}`}    onClick={() => { setTab('login');    setErr(''); setOk(''); }}>Sign In</button>
            <button className={`auth-tab ${tab==='register'?'active':''}`} onClick={() => { setTab('register'); setErr(''); setOk(''); }}>Create Account</button>
          </div>

          {tab === 'login' ? (
            <div className="fade-in" key="login">
              <h2 className="auth-form-title">Welcome <em>back</em>.</h2>
              <p className="auth-form-sub">Sign in to continue your session.</p>
              {err && <div className="auth-alert err">{err}</div>}
              {ok  && <div className="auth-alert ok">{ok}</div>}
              <div className="auth-field">
                <label>Email</label>
                <input type="email" placeholder="name@organization.com" value={form.email}
                       onChange={set('email')} className={err && !form.email ? 'error':''}
                       onKeyDown={e => e.key==='Enter' && login()} />
              </div>
              <div className="auth-field">
                <label>Password</label>
                <input type="password" placeholder="••••••••" value={form.password}
                       onChange={set('password')} className={err && !form.password ? 'error':''}
                       onKeyDown={e => e.key==='Enter' && login()} />
              </div>
              <button className="auth-submit" onClick={login} disabled={loading} data-magnet>
                {loading ? <><div className="auth-spinner"/><span>Authenticating</span></> : <span>Sign In →</span>}
              </button>
              <div className="auth-hint">Create an account, or configure an admin through environment variables.</div>
            </div>
          ) : (
            <div className="fade-in" key="register">
              <h2 className="auth-form-title">Create an <em>account</em>.</h2>
              <p className="auth-form-sub">Register to access SignTurk.</p>
              {err && <div className="auth-alert err">{err}</div>}
              {ok  && <div className="auth-alert ok">{ok}</div>}
              <div className="auth-field"><label>Full name</label><input type="text" placeholder="Jane Doe" value={form.full_name} onChange={set('full_name')} /></div>
              <div className="auth-field"><label>Email</label><input type="email" placeholder="name@organization.com" value={form.email} onChange={set('email')} /></div>
              <div className="auth-field"><label>Password</label><input type="password" placeholder="Choose a password" value={form.password} onChange={set('password')} /></div>
              <div className="auth-field"><label>Confirm Password</label><input type="password" placeholder="Repeat password" value={form.confirm} onChange={set('confirm')} onKeyDown={e => e.key==='Enter' && register()} /></div>
              <button className="auth-submit" onClick={register} disabled={loading} data-magnet>
                {loading ? <><div className="auth-spinner"/><span>Processing</span></> : <span>Create Account →</span>}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   SIDEBAR + TOPBAR
   ============================================================ */
function Sidebar({ route, navigate, user, onLogout }){
  const isAdmin = route.startsWith('admin/');
  const canAdmin = user?.role === 'Admin';
  const userNav = [
    ['dashboard','Dashboard','D'],
    ['live','Live Translation','L'],
    ['avatar','Avatar Studio','A'],
    ['avatar3d','3D Animation','3'],
    ['history','History','H'],
    ['settings','Settings','S'],
  ];
  const adminNav = [
    ['admin/overview','Overview','O'],
    ['admin/users','Users','U'],
    ['admin/logs','Activity Logs','L'],
    ['admin/model','AI Monitor','M'],
    ['admin/dictionary','Dictionary','D'],
    ['admin/system','System','Y'],
  ];
  const nav = isAdmin ? adminNav : userNav;
  const initials = (user?.full_name || 'U').split(' ').map(w => w[0]).join('').toUpperCase().slice(0,2);

  return (
    <aside className="sidebar">
      <div className="brand" onClick={() => navigate(isAdmin ? 'admin/overview' : 'dashboard')}>
        <div className="brand-mark"><span>S</span></div>
        <div className="brand-text">
          <h1>SignTurk</h1>
          <p>Sign Language AI</p>
        </div>
      </div>

      <div className="user-chip">
        <div className="user-chip-avatar">{initials}</div>
        <div className="user-chip-text">
          <strong>{user?.full_name || 'User'}</strong>
          <span>{user?.role || 'User'}</span>
        </div>
        <button className="user-chip-exit" onClick={onLogout} title="Sign out">↗</button>
      </div>

      <div>
        <div className="nav-group-title">{isAdmin ? 'Administration' : 'Workspace'}</div>
        <div className="nav-list">
          {nav.map(([id, label, mark]) => (
            <button key={id} className={route === id ? 'active' : ''} onClick={() => navigate(id)}>
              <span className="nav-icon">{mark}</span>
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="sidebar-bottom">
        {canAdmin && (
          <button className="mode-toggle" onClick={() => navigate(isAdmin ? 'dashboard' : 'admin/overview')}>
            {isAdmin ? '↩ User View' : '↗ Admin View'}
          </button>
        )}
        <div className="status-card">
          <div className="status-row"><span>Model</span><span className="status-dot"><i/>Online</span></div>
          <div className="status-row"><span>Pipeline</span><span className="mono" style={{fontSize:10,color:'var(--ink-3)'}}>Ready</span></div>
          <div className="status-row"><span>Version</span><span className="mono" style={{fontSize:10,color:'var(--ink-3)'}}>v2.2</span></div>
        </div>
      </div>
    </aside>
  );
}

function Topbar({ title, subtitle, user, route }){
  const initials = (user?.full_name || 'U').split(' ').map(w => w[0]).join('').toUpperCase().slice(0,2);
  const isAdmin = route.startsWith('admin/');
  return (
    <div className="topbar">
      <div className="topbar-left">
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      <div className="topbar-right">
        <div className="chip live"><i/>{isAdmin ? 'Admin Mode' : 'Live'}</div>
        <div className="chip mono">v2.2</div>
        <div className="avatar">{initials}</div>
      </div>
    </div>
  );
}

/* ============================================================
   SHARED COMPONENTS
   ============================================================ */
function Stat({ label, value, sub, tag, emphasized }){
  return (
    <div className="card stat hover">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',gap:10}}>
        <div style={{flex:1,minWidth:0}}>
          <div className="stat-label">{label}</div>
          <div className="stat-value">{emphasized ? <>{value}<em>.</em></> : value}</div>
          {sub && <div className="stat-sub">{sub}</div>}
        </div>
        {tag && <span className={`tag ${tag.color}`}>{tag.text}</span>}
      </div>
    </div>
  );
}

function LineChart(){
  return (
    <svg viewBox="0 0 800 180" preserveAspectRatio="none">
      <defs>
        <linearGradient id="lineG" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(201,178,137,.32)"/>
          <stop offset="100%" stopColor="rgba(201,178,137,0)"/>
        </linearGradient>
        <linearGradient id="lineStroke" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#c9b289"/>
          <stop offset="100%" stopColor="#7b95b3"/>
        </linearGradient>
      </defs>
      <path d="M 40 150 C 120 138, 160 100, 240 108 S 360 50, 430 78 S 560 34, 640 68 S 710 60, 760 38" fill="none" stroke="url(#lineStroke)" strokeWidth="1.8" strokeLinecap="round"/>
      <path d="M 40 150 C 120 138, 160 100, 240 108 S 360 50, 430 78 S 560 34, 640 68 S 710 60, 760 38 L 760 180 L 40 180 Z" fill="url(#lineG)"/>
      {[40,240,430,640,760].map((x,i) => <circle key={i} cx={x} cy={[150,108,78,68,38][i]} r="3" fill="#c9b289"/>)}
    </svg>
  );
}

function Toast({ message, onDone }){
  useEffect(() => { const t = setTimeout(onDone, 2400); return () => clearTimeout(t); }, [onDone]);
  return <div className="toast ok"><em>✓</em>{message}</div>;
}

/* ============================================================
   DASHBOARD
   ============================================================ */
function DashboardPage({ navigate }){
  const [ov, setOv]           = useState(null);
  const [model, setModel]     = useState(null);
  const [history, setHistory] = useState([]);
  useEffect(() => {
    authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.adminOverview).then(r => r.json()).then(setOv).catch(() => {});
    authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.adminModel).then(r => r.json()).then(setModel).catch(() => {});
    authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.history).then(r => r.json()).then(setHistory).catch(() => {});
  }, []);
  const conf = ov ? ov.avg_confidence : 0;

  return (
    <div className="page page-enter"><div className="page-container">
      <div className="hero-row">
        <div className="hero-banner">
          <div className="hero-banner-inner">
            <span className="eyebrow">AI-Powered Platform</span>
            <h3>Real-time Turkish Sign Language <em>recognition</em>.</h3>
            <p>A graduation-project pipeline running BiLSTM with Temporal Attention across 179 sign classes. Open the live canvas to start translating.</p>
            <div className="hero-banner-actions">
              <button className="btn primary" data-magnet onClick={() => navigate('live')}>Open Live Translation →</button>
              <button className="btn ghost" onClick={() => navigate('avatar')}>Avatar Studio</button>
            </div>
          </div>
        </div>
        <div className="stack">
          <div className="card hover">
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
              <div>
                <div className="label">Average Confidence</div>
                <div className="stat-value" style={{marginTop:8,fontSize:40}}>{conf}<em>%</em></div>
              </div>
              <span className={`tag ${conf>=70?'green':conf>=40?'amber':'red'}`}>{conf>=70?'Healthy':conf>=40?'Moderate':'Low'}</span>
            </div>
            <div className="progress" style={{marginTop:12}}><i style={{width:`${conf}%`}}/></div>
          </div>
          <div className="card hover">
            <div className="label">Model Capacity</div>
            <div style={{marginTop:8,display:'flex',alignItems:'baseline',gap:8}}>
              <span className="stat-value" style={{fontSize:36}}>{model?model.num_classes:'—'}<em>.</em></span>
              <span style={{fontSize:13,color:'var(--ink-muted)'}}>sign classes</span>
            </div>
            <div className="stat-sub">{model ? model.model_version : 'BiLSTM + Temporal Attention · 156 features'}</div>
          </div>
        </div>
      </div>

      <div className="grid-4" style={{marginBottom:18}}>
        <Stat label="Total Predictions" value={ov?String(ov.translations):'—'} sub="Stored in database"      tag={{text:'Live',  color:'gold'}} />
        <Stat label="Inference Time"    value={model?model.avg_inference:'—'}  sub="Per 16-frame window"    tag={{text:'Normal',color:'green'}} />
        <Stat label="Dictionary"        value={model?String(model.label_map_size):'—'} sub="Sign mappings loaded" tag={{text:'Synced',color:'blue'}} />
        <Stat label="Users"             value={ov?String(ov.total_users):'—'} sub={`${ov?ov.active_users:'—'} active`} tag={{text:'Active',color:'green'}} />
      </div>

      <div className="row-split">
        <div className="card">
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
            <div>
              <div className="card-title">System Status</div>
              <div className="card-sub">Live readiness across the platform</div>
            </div>
            <span className="tag green">Operational</span>
          </div>
          <div className="stack" style={{gap:8}}>
            {[
              ['Model',     model?.model_loaded     ? 'Loaded and operational'  : 'Not loaded',     model?.model_loaded     ? 'green':'red'],
              ['MediaPipe', model?.mediapipe_ready  ? 'Hand detection active'    : 'Unavailable',   model?.mediapipe_ready  ? 'green':'red'],
              ['Database',  model?.database || 'Database ready', 'green'],
              ['WebSocket', 'Ready for live streams',        'blue'],
            ].map(([t,d,c]) => (
              <div className="list-row" key={t}>
                <div className="mini-icon gold">{t.substring(0,2).toUpperCase()}</div>
                <div><h4>{t}</h4><p>{d}</p></div>
                <span className={`tag ${c}`}>{c==='green'?'OK':c==='red'?'Error':'Info'}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">Recent Translations</div>
          <div className="card-sub">Last few confident predictions</div>
          {history.length === 0 ? (
            <div style={{padding:'20px 0',fontSize:13,color:'var(--ink-muted)',textAlign:'center'}}>No history yet. Start a live session.</div>
          ) : (
            <div className="stack" style={{gap:8}}>
              {history.slice(0, 5).map(row => (
                <div className="list-row" key={row.id}>
                  <div className="mini-icon">{(row.mode||'LV').slice(0,2).toUpperCase()}</div>
                  <div><h4>{row.result}</h4><p>{row.time}</p></div>
                  <span className="tag gold">{row.conf}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div></div>
  );
}

/* ============================================================
   LIVE TRANSLATION
   ============================================================ */
function LivePage(){
  const [liveResult, setLiveResult] = useState(null);
  const [status, setStatus]         = useState('Idle');
  const [connected, setConnected]   = useState(false);
  const [recent, setRecent]         = useState([]);
  // ── ✓/✗ approval + sentence strip ──────────────────────────────
  const [pending, setPending]             = useState(null);  // {word,label_en}
  const [sentenceWords, setSentenceWords] = useState([]);    // approved words
  const [sentence, setSentence]           = useState('');    // assembled sentence
  const wsRef       = useRef(null);
  const videoRef    = useRef(null);
  const streamRef   = useRef(null);
  const canvasRef   = useRef(null);
  const intervalRef = useRef(null);
  const inFlightRef = useRef(false);
  useMagnets();

  const captureFrame = () => {
    const v = videoRef.current, c = canvasRef.current;
    if (!v || !c || v.videoWidth === 0) return null;
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    return c.toDataURL('image/jpeg', 0.7);
  };

  const startCamera = async () => {
    try{
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = s;
      if (videoRef.current) videoRef.current.srcObject = s;
      setStatus('Camera active');
    } catch { setStatus('Camera access denied'); }
  };

  const stopCamera = () => {
    if (streamRef.current){ streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  const startLive = async () => {
    await startCamera();
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) return;
    setStatus('Connecting…');
    const stored = JSON.parse(localStorage.getItem('tsl_user') || 'null');
    const token = stored?.access_token ? `?token=${encodeURIComponent(stored.access_token)}` : '';
    wsRef.current = new WebSocket(window.TSL_API.wsUrl + window.TSL_API.endpoints.live + token);
    wsRef.current.onopen = () => {
      setConnected(true); setStatus('Connected');
      intervalRef.current = setInterval(() => {
        if (inFlightRef.current) return;
        const f = captureFrame();
        if (f && wsRef.current?.readyState === WebSocket.OPEN) {
          inFlightRef.current = true;
          wsRef.current.send(f);
        }
      }, 250);
    };
    wsRef.current.onmessage = (e) => {
      inFlightRef.current = false;
      const d = JSON.parse(e.data);
      // ── approval / sentence-strip control replies ───────────────
      if (d.type === 'sentence_update'){
        setSentenceWords(d.words || []);
        setSentence(d.sentence || '');
        setPending(null);
        return;
      }
      if (d.type === 'rejected'){ setPending(null); return; }
      if (d.class_id === -1){ setStatus(d.label_tr || 'Processing…'); return; }
      setLiveResult(d);
      // Only confident signs become an approval candidate the user can confirm.
      if (d.above_threshold && d.pending_word){
        setPending({ word: d.pending_word, label_en: d.label_en });
      }
      if (d.confidence >= 0.4) setRecent(p => [d, ...p].slice(0, 8));
      setStatus('Recognizing');
    };
    wsRef.current.onerror = () => { inFlightRef.current = false; setStatus('Connection error'); };
    wsRef.current.onclose = () => { inFlightRef.current = false; setConnected(false); setStatus('Disconnected'); };
  };

  const stopLive = () => {
    if (intervalRef.current){ clearInterval(intervalRef.current); intervalRef.current = null; }
    inFlightRef.current = false;
    if (wsRef.current){ wsRef.current.close(); wsRef.current = null; }
    stopCamera();
    setConnected(false);
    setStatus('Idle');
    setPending(null);
  };

  // ── ✓/✗ approval controls ──────────────────────────────────────
  const sendAction = (obj) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  };
  // "Doğru": persist the word and append it to the sentence strip.
  const approveWord  = () => { if (pending) sendAction({ action:'approve', word: pending.word }); };
  // "Yanlış": discard the candidate, nothing is saved.
  const rejectWord   = () => { setPending(null); sendAction({ action:'reject' }); };
  const undoWord     = () => sendAction({ action:'undo' });
  const resetSentence= () => sendAction({ action:'reset_sentence' });

  useEffect(() => () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    inFlightRef.current = false;
    if (wsRef.current) wsRef.current.close();
    stopCamera();
  }, []);

  const pct = liveResult ? Math.round(liveResult.confidence * 100) : 0;
  const ringR = 32;
  const ringC = 2 * Math.PI * ringR;
  const ringOffset = ringC - (pct / 100) * ringC;

  return (
    <div className="page page-enter"><div className="page-container">
      <div className="live-shell">
        <div className="camera-stage">
          <div className="camera-frame">
            <div className="camera-grid-bg" />
            <video ref={videoRef} autoPlay playsInline muted style={{opacity: streamRef.current ? 1 : 0, transition:'opacity .5s var(--ease)'}} />
            <canvas ref={canvasRef} style={{display:'none'}} />
            <div className="camera-overlay">
              <div className="camera-vignette" />
              {connected && <div className="camera-scan" />}
              <div className="camera-corners"><i/><i/><i/><i/></div>
            </div>
            {!streamRef.current && (
              <div className="camera-center">
                <div className="camera-placeholder">
                  <div className="big">N</div>
                  <h4>Camera idle</h4>
                  <p>Click <strong style={{color:'var(--champagne)'}}>Start</strong> to activate your camera and begin live sign recognition.</p>
                </div>
              </div>
            )}
            <div className="camera-hud">
              <div className={`hud-pill ${connected?'on':''}`}><i/>{connected?'LIVE':'OFFLINE'}</div>
              <div className="hud-pill">ws · 250ms · 16 frames</div>
            </div>
            <div className="camera-footer">
              <div className="camera-controls">
                {!connected ? (
                  <button className="ctrl-btn go" onClick={startLive} data-magnet>● Start Recognition</button>
                ) : (
                  <button className="ctrl-btn stop" onClick={stopLive} data-magnet>■ Stop</button>
                )}
              </div>
              {connected && (
                <div className="wave-mini">
                  {Array.from({length:7}).map((_,i) => <i key={i}/>)}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="stack">
          <div className="result-card">
            <div className="result-card-inner">
              <div className="label">Detected Sign</div>
              <div className="result-big">{liveResult?.label_tr ? <>{liveResult.label_tr}<em>.</em></> : '—'}</div>
              <div className="result-en">{liveResult?.label_en || status}</div>
              <div className="result-meta">
                <div>
                  <div className="label" style={{marginBottom:4}}>Confidence</div>
                  <div className="mono" style={{fontSize:12,color:'var(--ink-3)'}}>class_id · {liveResult?.class_id ?? '—'}</div>
                </div>
                <div className="result-conf-ring">
                  <svg viewBox="0 0 80 80">
                    <defs>
                      <linearGradient id="confGrad" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0%"   stopColor="#c9b289"/>
                        <stop offset="100%" stopColor="#7b95b3"/>
                      </linearGradient>
                    </defs>
                    <circle className="bg" cx="40" cy="40" r={ringR}/>
                    <circle className="fg" cx="40" cy="40" r={ringR} strokeDasharray={ringC} strokeDashoffset={ringOffset}/>
                  </svg>
                  <div className="pct">{pct}<em>%</em></div>
                </div>
              </div>

              {/* ── ✓/✗ approval: only confirmed words are saved ── */}
              {pending && connected && (
                <div style={{marginTop:16}}>
                  <div className="label" style={{marginBottom:8}}>
                    “{pending.word}” doğru mu?
                  </div>
                  <div style={{display:'flex',gap:10}}>
                    <button onClick={approveWord} data-magnet
                      style={{flex:1,padding:'12px 14px',border:'none',borderRadius:12,cursor:'pointer',
                        fontWeight:600,fontSize:14,color:'#0c2a16',
                        background:'linear-gradient(135deg,#7fd99a,#3fae6a)'}}>
                      ✓ Doğru
                    </button>
                    <button onClick={rejectWord} data-magnet
                      style={{flex:1,padding:'12px 14px',borderRadius:12,cursor:'pointer',
                        fontWeight:600,fontSize:14,color:'#e0a3a3',
                        border:'1px solid rgba(200,80,80,.4)',background:'rgba(200,80,80,.08)'}}>
                      ✗ Yanlış
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── Cümle Şeridi (sentence strip) ── */}
          <div className="card">
            <div className="card-title">Cümle Şeridi</div>
            <div className="card-sub">Onaylanan kelimeler · {sentenceWords.length}</div>
            {sentenceWords.length === 0 ? (
              <div style={{padding:'14px 0',fontSize:12,color:'var(--ink-muted)'}}>
                Bir işaret algılandığında <strong style={{color:'var(--champagne)'}}>✓ Doğru</strong> ile kelimeyi şeride ekleyin.
              </div>
            ) : (
              <>
                <div style={{display:'flex',flexWrap:'wrap',gap:6,marginBottom:12}}>
                  {sentenceWords.map((w,i) => (
                    <span key={i} style={{padding:'5px 11px',borderRadius:999,fontSize:13,color:'var(--ink)',
                      background:'rgba(201,178,137,.14)',border:'1px solid rgba(201,178,137,.32)'}}>{w}</span>
                  ))}
                </div>
                <div className="serif" style={{fontSize:19,color:'var(--ink)',lineHeight:1.4}}>{sentence || '—'}</div>
                <div style={{display:'flex',gap:8,marginTop:14}}>
                  <button onClick={undoWord}
                    style={{padding:'7px 12px',borderRadius:10,cursor:'pointer',fontSize:12,
                      border:'1px solid var(--line)',background:'transparent',color:'var(--ink-3)'}}>↶ Geri al</button>
                  <button onClick={resetSentence}
                    style={{padding:'7px 12px',borderRadius:10,cursor:'pointer',fontSize:12,
                      border:'1px solid var(--line)',background:'transparent',color:'var(--ink-3)'}}>Temizle</button>
                </div>
              </>
            )}
          </div>

          <div className="card">
            <div className="card-title">Top Predictions</div>
            <div className="card-sub">Ranked by softmax confidence</div>
            {liveResult?.top_predictions?.length > 0 ? (
              <div className="topk">
                {liveResult.top_predictions.map((p, i) => (
                  <div className="topk-row" key={i}>
                    <span className="rank">{['I','II','III','IV','V'][i]}</span>
                    <div className="words"><strong>{p.label_tr}</strong><span>{p.label_en}</span></div>
                    <span className="pct">{(p.confidence*100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{padding:'14px 0',fontSize:12,color:'var(--ink-muted)'}}>Predictions will appear here once signing begins.</div>
            )}
          </div>

          <div className="card">
            <div className="card-title">Recent Session</div>
            <div className="card-sub">{recent.length} confident prediction{recent.length===1?'':'s'}</div>
            {recent.length === 0 ? (
              <div style={{padding:'14px 0',fontSize:12,color:'var(--ink-muted)'}}>No confident detections yet.</div>
            ) : (
              <div className="stack" style={{gap:6}}>
                {recent.map((item, i) => (
                  <div className="list-row" key={i}>
                    <div className="mini-icon">{String(item.class_id).padStart(2,'0')}</div>
                    <div><h4>{item.label_tr}</h4><p>{item.label_en}</p></div>
                    <span className="tag gold">{Math.round(item.confidence*100)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div></div>
  );
}

/* ============================================================
   AVATAR STUDIO
   ============================================================ */

/* ============================================================
   3D ANİMASYON SAYFASI
   ============================================================ */
function Avatar3DPage(){
  const [signs, setSigns] = React.useState([]);
  const [search, setSearch] = React.useState('');
  const [selected, setSelected] = React.useState('');
  const iframeRef = React.useRef(null);

  React.useEffect(() => {
    authFetch('/signs')
      .then(r => r.json())
      .then(d => { setSigns(d.words || []); })
      .catch(() => {});
  }, []);

  const filtered = signs.filter(w => w.toLowerCase().includes(search.toLowerCase()));

  const playSign = (word) => {
    setSelected(word);
    if (iframeRef.current) {
      iframeRef.current.contentWindow.postMessage({ type: 'play_sign', word }, '*');
    }
  };

  return (
    <div className="page page-enter"><div className="page-container">
      <div className="work-grid">

        {/* Sol: 3D görüntüleyici iframe */}
        <div className="card" style={{padding:0,overflow:'hidden',minHeight:520,display:'flex',flexDirection:'column',position:'relative'}}>
          <div style={{padding:'18px 22px',borderBottom:'1px solid var(--line)',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
            <div>
              <div className="card-title">3D İşaret Animasyonu</div>
              <div className="card-sub">{selected ? `Oynatılıyor: ${selected}` : 'Sağdan bir işaret seçin'}</div>
            </div>
            {selected && (
              <span className="tag gold mono">{selected}</span>
            )}
          </div>
          <iframe
            ref={iframeRef}
            src="/avatar3d"
            style={{flex:1,border:'none',width:'100%',minHeight:460,background:'var(--pearl-warm)'}}
            title="3D Animasyon"
            allow="camera; microphone"
          />
        </div>

        {/* Sağ: İşaret listesi */}
        <div className="stack">
          <div className="card">
            <div className="card-title" style={{marginBottom:4}}>İşaret Listesi</div>
            <div className="card-sub" style={{marginBottom:12}}>{signs.length} işaret yüklendi</div>
            <input
              className="search"
              placeholder="İşaret ara…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{marginBottom:12}}
            />
            <div className="stack" style={{gap:6,maxHeight:480,overflowY:'auto',paddingRight:4}}>
              {filtered.slice(0,120).map(word => (
                <div
                  className={"list-row link" + (selected===word ? " active" : "")}
                  key={word}
                  onClick={() => playSign(word)}
                  style={selected===word ? {background:'var(--gold-soft)',border:'1px solid var(--gold-edge)'} : {}}
                >
                  <div className="mini-icon" style={{background:selected===word?'var(--champagne)':'var(--frost)',color:selected===word?'#fff':'var(--ink-3)',fontSize:10}}>▶</div>
                  <div><h4 style={{textTransform:'capitalize'}}>{word}</h4></div>
                  <span className="tag neutral">Oynat</span>
                </div>
              ))}
              {filtered.length === 0 && (
                <div style={{textAlign:'center',padding:40,color:'var(--ink-muted)'}}>Sonuç bulunamadı</div>
              )}
            </div>
          </div>
        </div>

      </div>
    </div></div>
  );
}

function AvatarPage(){
  const [rows, setRows]       = useState([]);
  const [text, setText]       = useState('');
  const [selected, setSelected] = useState(null);
  const [search, setSearch]   = useState('');

  useEffect(() => {
    authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.dictionary).then(r => r.json()).then(d => {
      setRows(d); if (d.length) setSelected(d[0]);
    }).catch(() => {});
  }, []);

  const lookup = () => {
    const m = rows.find(r => r.tr.toLowerCase() === text.trim().toLowerCase() || r.en.toLowerCase() === text.trim().toLowerCase());
    if (m) setSelected(m);
    else setSelected({ classId:'—', tr: text || '—', en:'not found', animation:'unknown_sign', category:'—' });
  };

  const filtered = rows.filter(r => r.tr.toLowerCase().includes(search.toLowerCase()) || r.en.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="page page-enter"><div className="page-container">
      <div className="work-grid">
        <div className="card" style={{minHeight:480,display:'flex',flexDirection:'column',justifyContent:'center',alignItems:'center',background:'linear-gradient(135deg,var(--pearl-warm) 0%,var(--frost) 100%)',position:'relative',overflow:'hidden'}}>
          <div style={{position:'absolute',top:'-30%',right:'-20%',width:400,height:400,background:'radial-gradient(circle,rgba(201,178,137,.20),transparent 70%)',borderRadius:'50%',filter:'blur(60px)',pointerEvents:'none'}}/>
          <div style={{position:'absolute',bottom:'-30%',left:'-20%',width:380,height:380,background:'radial-gradient(circle,rgba(123,149,179,.16),transparent 70%)',borderRadius:'50%',filter:'blur(60px)',pointerEvents:'none'}}/>
          <div style={{position:'relative',zIndex:1,textAlign:'center',maxWidth:480,padding:'0 32px'}}>
            <div className="eyebrow">Avatar Studio</div>
            <h3 className="serif" style={{fontSize:42,lineHeight:1.08,marginTop:14,color:'var(--ink)',fontWeight:300,letterSpacing:'-.02em'}}>Type a sign.<br/>Retrieve its <em>animation</em>.</h3>
            <p style={{fontSize:14,color:'var(--ink-3)',lineHeight:1.7,marginTop:14,marginBottom:24}}>Every sign in the Turkish Sign Language dictionary maps to an avatar animation key. Look up words to see their mapping.</p>
            <div style={{display:'flex',gap:8}}>
              <input className="search" style={{flex:1}} value={text} onChange={e => setText(e.target.value)} placeholder="e.g. Merhaba or hello" onKeyDown={e => e.key==='Enter' && lookup()} />
              <button className="btn primary" data-magnet onClick={lookup}>Look Up →</button>
            </div>
            {selected && (
              <div style={{marginTop:24,padding:20,background:'var(--pearl-2)',border:'1px solid var(--line)',borderRadius:14,textAlign:'left',boxShadow:'var(--shadow-1)'}}>
                <div className="label">Result</div>
                <div className="serif" style={{fontSize:36,letterSpacing:'-.02em',marginTop:6,color:'var(--ink)',fontWeight:300}}>{selected.tr} <span style={{color:'var(--ink-muted)',fontSize:16,fontStyle:'italic',fontWeight:400}}>· {selected.en}</span></div>
                <div style={{marginTop:10,display:'flex',gap:10,flexWrap:'wrap'}}>
                  <span className="tag blue mono">class_id: {selected.classId}</span>
                  <span className="tag gold mono">{selected.animation}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="stack">
          <div className="card">
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14}}>
              <div>
                <div className="card-title">Dictionary</div>
                <div className="card-sub">{rows.length} sign classes loaded</div>
              </div>
            </div>
            <input className="search" placeholder="Filter signs…" value={search} onChange={e => setSearch(e.target.value)} style={{marginBottom:12}} />
            <div className="stack" style={{gap:6,maxHeight:440,overflowY:'auto',paddingRight:4}}>
              {filtered.slice(0, 80).map(row => (
                <div className="list-row link" key={row.classId} onClick={() => { setSelected(row); setText(row.tr); }}>
                  <div className="mini-icon">{String(row.classId).padStart(3,'0')}</div>
                  <div><h4>{row.tr}</h4><p>{row.en}</p></div>
                  <span className="tag neutral">Select</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div></div>
  );
}

/* ============================================================
   HISTORY
   ============================================================ */
function HistoryPage(){
  const [rows, setRows]       = useState([]);
  const [search, setSearch]   = useState('');
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.history)
      .then(r => r.json()).then(d => { setRows(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);
  const filtered = rows.filter(r => !search || (r.result||'').toLowerCase().includes(search.toLowerCase()) || (r.id||'').toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="page page-enter"><div className="page-container">
      <div className="card">
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:18,gap:14,flexWrap:'wrap'}}>
          <div>
            <div className="card-title">Translation History</div>
            <div className="card-sub">{rows.length} record{rows.length===1?'':'s'} · sorted by most recent</div>
          </div>
          <input className="search" placeholder="Search by sign or session…" value={search} onChange={e => setSearch(e.target.value)} style={{maxWidth:320}} />
        </div>

        {loading ? (
          <div className="stack" style={{gap:8}}>
            {Array.from({length:4}).map((_,i) => <div key={i} className="skeleton" style={{height:52}}/>)}
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <div className="big">∅</div>
            <h4>No history yet</h4>
            <p>Confident predictions from Live Translation will appear here. Start a session to begin building your history.</p>
          </div>
        ) : (
          <table className="table">
            <thead><tr><th>Session</th><th>Mode</th><th>Result</th><th>Confidence</th><th>Timestamp</th></tr></thead>
            <tbody>
              {filtered.map(row => (
                <tr key={row.id}>
                  <td className="mono" style={{fontSize:11}}>{row.id}</td>
                  <td><span className="tag neutral">{row.mode}</span></td>
                  <td className="strong">{row.result}</td>
                  <td><span className="tag gold">{row.conf}</span></td>
                  <td className="mono" style={{fontSize:11,color:'var(--ink-muted)'}}>{row.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div></div>
  );
}

/* ============================================================
   SETTINGS
   ============================================================ */
function SettingsPage(){
  const [settings, setSettings] = useState({ camera:'Default Camera', voice:'Female Voice', speech_rate:1.0, avatar_speed:1.0, tts_enabled:true, notifications_enabled:true, avatar_enabled:true, websocket_enabled:true });
  const [toast, setToast]       = useState(null);
  useMagnets();

  useEffect(() => {
    authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.settings).then(r => r.json()).then(d => setSettings(prev => ({ ...prev, ...d }))).catch(() => {});
  }, []);

  const save = async () => {
    try{
      await authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.settings, {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(settings)
      });
      setToast('Settings saved');
    } catch { setToast('Failed to save'); }
  };

  const flip   = (k)    => setSettings(p => ({ ...p, [k]: !p[k] }));
  const setVal = (k, v) => setSettings(p => ({ ...p, [k]: v }));

  return (
    <div className="page page-enter"><div className="page-container">
      <div className="grid-2">
        <div>
          <div className="settings-section">
            <h3>Runtime <em style={{color:'var(--gold-deep)',fontStyle:'italic',fontWeight:400}}>configuration</em></h3>
            <p>Tune camera, voice, and pipeline behavior</p>
          </div>
          <div className="card">
            <div className="field-row">
              <div className="field">
                <label>Camera source</label>
                <select value={settings.camera} onChange={e => setVal('camera', e.target.value)}>
                  <option>Default Camera</option><option>External Camera</option>
                </select>
              </div>
              <div className="field">
                <label>TTS voice</label>
                <select value={settings.voice} onChange={e => setVal('voice', e.target.value)}>
                  <option>Female Voice</option><option>Male Voice</option>
                </select>
              </div>
            </div>
            <div style={{marginTop:16}}>
              <div className="slider-row">
                <div className="meta"><strong>Speech rate</strong><span>Controls TTS playback speed</span></div>
                <input type="range" min="0.5" max="2" step="0.1" value={settings.speech_rate} onChange={e => setVal('speech_rate', parseFloat(e.target.value))} />
                <div className="val">{settings.speech_rate.toFixed(1)}x</div>
              </div>
              <div className="slider-row">
                <div className="meta"><strong>Avatar speed</strong><span>Animation playback multiplier</span></div>
                <input type="range" min="0.5" max="2" step="0.1" value={settings.avatar_speed} onChange={e => setVal('avatar_speed', parseFloat(e.target.value))} />
                <div className="val">{settings.avatar_speed.toFixed(1)}x</div>
              </div>
            </div>
            <div style={{marginTop:18,display:'flex',justifyContent:'flex-end'}}>
              <button className="btn gold" onClick={save} data-magnet>Save Configuration →</button>
            </div>
          </div>
        </div>

        <div>
          <div className="settings-section">
            <h3>Feature <em style={{color:'var(--gold-deep)',fontStyle:'italic',fontWeight:400}}>toggles</em></h3>
            <p>Enable or disable platform capabilities</p>
          </div>
          <div className="card">
            {[
              ['tts_enabled',           'Text-to-Speech',           'Spoken output for detected signs'],
              ['notifications_enabled', 'Desktop notifications',    'System alerts for new predictions'],
              ['avatar_enabled',        'Avatar playback',          'Render matching sign animations'],
              ['websocket_enabled',     'WebSocket mode',           'Real-time bidirectional inference'],
            ].map(([k,t,d]) => (
              <div className="toggle-row" key={k}>
                <div className="meta"><strong>{t}</strong><span>{d}</span></div>
                <div className={`toggle ${settings[k]?'on':''}`} onClick={() => flip(k)}><span/></div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {toast && <Toast message={toast} onDone={() => setToast(null)} />}
    </div></div>
  );
}

/* ============================================================
   ADMIN PAGES
   ============================================================ */
function AdminOverviewPage(){
  const [ov, setOv]     = useState(null);
  const [logs, setLogs] = useState([]);
  useEffect(() => {
    authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.adminOverview).then(r => r.json()).then(setOv).catch(() => {});
    authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.adminLogs).then(r => r.json()).then(setLogs).catch(() => {});
  }, []);
  return (
    <div className="page page-enter"><div className="page-container">
      <div className="grid-4" style={{marginBottom:20}}>
        <Stat label="Total Users"     value={ov?String(ov.total_users):'—'}    sub="Registered accounts"   tag={{text:'DB',   color:'blue'}}  emphasized/>
        <Stat label="Active Users"    value={ov?String(ov.active_users):'—'}   sub="Currently active"       tag={{text:'Live', color:'green'}} emphasized/>
        <Stat label="Translations"    value={ov?String(ov.translations):'—'}   sub="All-time predictions"   tag={{text:'Total',color:'gold'}}  emphasized/>
        <Stat label="Avg. Confidence" value={ov?`${ov.avg_confidence}%`:'—'}   sub="Rolling model average"  tag={{text:'Model',color:'blue'}}/>
      </div>

      <div className="row-split">
        <div className="card">
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14}}>
            <div><div className="card-title">Platform Activity</div><div className="card-sub">Usage trend over time</div></div>
            <span className="tag gold">Real-time</span>
          </div>
          <div className="chart-wrap"><LineChart/></div>
        </div>

        <div className="card">
          <div className="card-title">Infrastructure</div>
          <div className="card-sub">Service readiness</div>
          <div className="stack" style={{gap:8}}>
            {[['Backend API','Response nominal','green'],['Model Pipeline','Inference operational','green'],['Dictionary','179 mappings loaded','blue'],['WebSocket','Ready for connections','green']].map(([t,d,c]) => (
              <div className="list-row" key={t}>
                <div className="mini-icon gold">{t.substring(0,2).toUpperCase()}</div>
                <div><h4>{t}</h4><p>{d}</p></div>
                <span className={`tag ${c}`}>OK</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid-2" style={{marginTop:20}}>
        <div className="card">
          <div className="card-title">Recent Events</div>
          <div className="card-sub">Latest activity across the platform</div>
          <div className="activity">
            {logs.length === 0 ? <div style={{fontSize:12,color:'var(--ink-muted)'}}>No events recorded.</div> : logs.slice(0, 8).map((log, i) => (
              <div className="activity-item" key={i}>
                <div className="activity-dot" style={{background: log.level==='Warning'?'var(--warn)':log.level==='Success'?'var(--success)':'var(--gold-deep)'}}/>
                <div><h5>{log.message}</h5><p>{log.user} · {log.level}</p></div>
                <time>{log.when}</time>
              </div>
            ))}
          </div>
        </div>
        <div className="card">
          <div className="card-title">API Reference</div>
          <div className="card-sub">Endpoints exposed by the backend</div>
          <pre className="mono" style={{fontSize:11,color:'var(--ink-3)',lineHeight:1.8,background:'var(--pearl)',border:'1px solid var(--line)',borderRadius:10,padding:16,whiteSpace:'pre-wrap',marginTop:8}}>{`GET  /api/health
GET  /api/admin/overview
GET  /api/admin/users
GET  /api/admin/logs
GET  /api/admin/model
GET  /api/history
GET  /api/dictionary
POST /api/auth/login
POST /api/auth/register
WS   /api/predict/live`}</pre>
        </div>
      </div>
    </div></div>
  );
}

function AdminUsersPage(){
  const [rows, setRows]     = useState([]);
  const [search, setSearch] = useState('');
  useEffect(() => { authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.adminUsers).then(r => r.json()).then(setRows).catch(() => {}); }, []);
  const filtered = rows.filter(u => !search || u.name.toLowerCase().includes(search.toLowerCase()) || u.email.toLowerCase().includes(search.toLowerCase()));
  return (
    <div className="page page-enter"><div className="page-container">
      <div className="card">
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14,gap:10,flexWrap:'wrap'}}>
          <div><div className="card-title">User Management</div><div className="card-sub">{rows.length} account{rows.length===1?'':'s'} · searchable registry</div></div>
          <input className="search" placeholder="Search users…" value={search} onChange={e => setSearch(e.target.value)} style={{maxWidth:300}} />
        </div>
        {filtered.length === 0 ? (
          <div className="empty-state"><div className="big">U</div><h4>No users found</h4><p>Try a different search term or check back after new accounts register.</p></div>
        ) : (
          <table className="table">
            <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Sessions</th><th>Status</th></tr></thead>
            <tbody>
              {filtered.map(u => (
                <tr key={u.email}>
                  <td className="strong">{u.name}</td>
                  <td className="mono" style={{fontSize:12}}>{u.email}</td>
                  <td><span className={`tag ${u.role==='Admin'?'gold':'neutral'}`}>{u.role}</span></td>
                  <td>{u.sessions}</td>
                  <td><span className={`tag ${u.status==='Active'?'green':'red'}`}>{u.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div></div>
  );
}

function AdminLogsPage(){
  const [rows, setRows] = useState([]);
  useEffect(() => { authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.adminLogs).then(r => r.json()).then(setRows).catch(() => {}); }, []);
  return (
    <div className="page page-enter"><div className="page-container">
      <div className="card">
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14}}>
          <div><div className="card-title">Activity Log</div><div className="card-sub">Chronological platform events</div></div>
          <span className="tag gold">{rows.length} events</span>
        </div>
        <div className="activity" style={{maxHeight:'unset'}}>
          {rows.length === 0 ? <div style={{padding:30,textAlign:'center',fontSize:13,color:'var(--ink-muted)'}}>No events recorded.</div> : rows.map((log, i) => (
            <div className="activity-item" key={i}>
              <div className="activity-dot" style={{background: log.level==='Warning'?'var(--warn)':log.level==='Success'?'var(--success)':'var(--gold-deep)'}}/>
              <div><h5>{log.message}</h5><p>{log.user} · {log.level}</p></div>
              <time>{log.when}</time>
            </div>
          ))}
        </div>
      </div>
    </div></div>
  );
}

function AdminModelPage(){
  const [m, setM] = useState(null);
  useEffect(() => { authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.adminModel).then(r => r.json()).then(setM).catch(() => {}); }, []);
  if (!m) return <div className="page page-enter"><div className="page-container"><div className="card" style={{color:'var(--ink-muted)'}}>Loading model information…</div></div></div>;
  return (
    <div className="page page-enter"><div className="page-container">
      <div className="grid-2">
        <div className="card">
          <div className="card-title">Model Configuration</div>
          <div className="card-sub">Runtime parameters and performance</div>
          <div className="stack" style={{gap:8}}>
            {[
              ['Version',         m.model_version,                 'gold'],
              ['Input Shape',     m.input_shape,                   'blue'],
              ['Output Shape',    m.output_shape,                  'blue'],
              ['Threshold',       String(m.confidence_threshold),  'amber'],
              ['Avg. Inference',  m.avg_inference,                 'green'],
              ['Classes',         String(m.num_classes),           'neutral'],
            ].map(([k,v,c]) => (
              <div className="list-row" key={k}>
                <div className="mini-icon gold">AI</div>
                <div><h4>{k}</h4><p className="mono" style={{fontSize:11}}>{v}</p></div>
                <span className={`tag ${c}`}>Tracked</span>
              </div>
            ))}
          </div>
        </div>

        <div className="stack">
          <div className="card">
            <div className="card-title">Architecture</div>
            <div className="card-sub">Pipeline summary</div>
            <p style={{fontSize:13.5,color:'var(--ink-3)',lineHeight:1.75,marginTop:8}}>Bidirectional LSTM (256, 128 units) with Temporal Attention mechanism. Input: 16-frame sequences of 156 hand landmark features. Output: 179 Turkish Sign Language classes via softmax.</p>
            <p style={{fontSize:13.5,color:'var(--ink-3)',lineHeight:1.75,marginTop:10}}>Preprocessing: relative coordinate normalization, finger joint angle computation, z-score standardization against training statistics.</p>
          </div>
          <div className="card">
            <div className="card-title">Health</div>
            <div className="card-sub">Live readiness</div>
            <div className="stack" style={{gap:8,marginTop:6}}>
              <div className="list-row"><div className="mini-icon gold">MD</div><div><h4>Model</h4><p>{m.model_loaded ? 'Loaded and ready' : 'Not loaded'}</p></div><span className={`tag ${m.model_loaded?'green':'red'}`}>{m.model_loaded?'OK':'Down'}</span></div>
              <div className="list-row"><div className="mini-icon gold">MP</div><div><h4>MediaPipe</h4><p>{m.mediapipe_ready ? 'Hand detection active' : 'Unavailable'}</p></div><span className={`tag ${m.mediapipe_ready?'green':'red'}`}>{m.mediapipe_ready?'OK':'Down'}</span></div>
              <div className="list-row"><div className="mini-icon gold">DB</div><div><h4>Database</h4><p>{m.database}</p></div><span className="tag green">OK</span></div>
            </div>
          </div>
        </div>
      </div>
    </div></div>
  );
}

function AdminDictionaryPage(){
  const [rows, setRows]     = useState([]);
  const [search, setSearch] = useState('');
  useEffect(() => { authFetch(window.TSL_API.baseUrl + window.TSL_API.endpoints.dictionary).then(r => r.json()).then(setRows).catch(() => {}); }, []);
  const filtered = rows.filter(r => r.tr.toLowerCase().includes(search.toLowerCase()) || r.en.toLowerCase().includes(search.toLowerCase()));
  return (
    <div className="page page-enter"><div className="page-container">
      <div className="card">
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:14,gap:10,flexWrap:'wrap'}}>
          <div><div className="card-title">Sign Dictionary</div><div className="card-sub">{rows.length} class mappings</div></div>
          <input className="search" placeholder="Filter by TR or EN…" value={search} onChange={e => setSearch(e.target.value)} style={{maxWidth:300}} />
        </div>
        <div style={{maxHeight:620,overflowY:'auto'}}>
          <table className="table">
            <thead><tr><th>ID</th><th>Turkish</th><th>English</th><th>Animation Key</th></tr></thead>
            <tbody>
              {filtered.map(row => (
                <tr key={row.classId}>
                  <td className="mono" style={{fontSize:11}}>{String(row.classId).padStart(3,'0')}</td>
                  <td className="strong">{row.tr}</td>
                  <td style={{color:'var(--ink-3)'}}>{row.en}</td>
                  <td className="mono" style={{fontSize:11,color:'var(--ink-muted)'}}>{row.animation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div></div>
  );
}

function AdminSystemPage(){
  return (
    <div className="page page-enter"><div className="page-container">
      <div className="grid-3">
        <div className="card hover">
          <div className="label">API Server</div>
          <div className="serif" style={{fontSize:24,marginTop:8,color:'var(--ink)',fontWeight:400,letterSpacing:'-.012em'}}>{window.TSL_API.baseUrl}</div>
          <div style={{marginTop:6,fontSize:12,color:'var(--ink-muted)'}}>FastAPI backend endpoint</div>
        </div>
        <div className="card hover">
          <div className="label">Database</div>
          <div className="serif" style={{fontSize:24,marginTop:8,color:'var(--ink)',fontWeight:400,letterSpacing:'-.012em'}}>SQLAlchemy <em style={{color:'var(--gold-deep)',fontStyle:'italic'}}>Database</em></div>
          <div style={{marginTop:6,fontSize:12,color:'var(--ink-muted)'}}>SQLite locally · PostgreSQL in hosted deployments</div>
        </div>
        <div className="card hover">
          <div className="label">Real-time Layer</div>
          <div className="serif" style={{fontSize:24,marginTop:8,color:'var(--ink)',fontWeight:400,letterSpacing:'-.012em'}}>WebSocket</div>
          <div style={{marginTop:6,fontSize:12,color:'var(--ink-muted)'}}>Live bidirectional inference stream</div>
        </div>
      </div>
    </div></div>
  );
}

/* ============================================================
   APP ROOT — splash, persistence, routing
   ============================================================ */
function App(){
  const [splash, setSplash] = useState(true);
  const [user, setUser]       = useStoredUser();
  const [route, navigate]     = useHashRoute();
  useMouseParallax();

  useEffect(() => { const t = setTimeout(() => setSplash(false), 1700); return () => clearTimeout(t); }, []);

  const AUTH_ROUTES = new Set(['home', 'auth']);
  const VALID = new Set(['dashboard','live','avatar','avatar3d','history','settings','admin/overview','admin/users','admin/logs','admin/model','admin/dictionary','admin/system']);

  useEffect(() => {
    if (!user && !AUTH_ROUTES.has(route)) {
      navigate('home');
    } else if (user && AUTH_ROUTES.has(route)) {
      navigate(user.role === 'Admin' ? 'admin/overview' : 'dashboard');
    } else if (user && !VALID.has(route)) {
      navigate(user.role === 'Admin' ? 'admin/overview' : 'dashboard');
    }
  }, [user, route]);

  useEffect(() => {
    if (splash) document.body.classList.remove('scrollable');
  }, [splash]);

  return (
    <>
      <CustomCursor />
      {splash && <Splash />}
      {(() => {
        if (!user){
          if (route === 'auth') return <AuthScreen onAuth={u => { setUser(u); navigate(u.role === 'Admin' ? 'admin/overview' : 'dashboard'); }} onBack={() => navigate('home')} />;
          return <LandingPage onLaunch={() => navigate('auth')} />;
        }
        const meta = {
          'dashboard':        ['Dashboard',         'Overview of your platform'],
          'live':             ['Live Translation',  'Real-time camera inference'],
          'avatar':           ['Avatar Studio',     'Sign-to-animation mapping'],
          'avatar3d':         ['3D Animation',      '3D işaret dili animasyonu'],
          'history':          ['Translation History','Past predictions archive'],
          'settings':         ['Settings',          'Runtime configuration'],
          'admin/overview':   ['Admin Overview',    'System and user intelligence'],
          'admin/users':      ['User Management',   'Accounts and sessions'],
          'admin/logs':       ['Activity Log',      'Event timeline'],
          'admin/model':      ['AI Monitor',        'Model performance and telemetry'],
          'admin/dictionary': ['Sign Dictionary',   'Class mapping registry'],
          'admin/system':     ['System',            'Infrastructure status'],
        }[route] || ['SignTurk',''];

        const page = (() => {
          switch(route){
            case 'dashboard':        return <DashboardPage navigate={navigate}/>;
            case 'live':             return <LivePage/>;
            case 'avatar':           return <AvatarPage/>;
            case 'avatar3d':         return <Avatar3DPage/>;
            case 'history':          return <HistoryPage/>;
            case 'settings':         return <SettingsPage/>;
            case 'admin/overview':   return <AdminOverviewPage/>;
            case 'admin/users':      return <AdminUsersPage/>;
            case 'admin/logs':       return <AdminLogsPage/>;
            case 'admin/model':      return <AdminModelPage/>;
            case 'admin/dictionary': return <AdminDictionaryPage/>;
            case 'admin/system':     return <AdminSystemPage/>;
            default:                 return <DashboardPage navigate={navigate}/>;
          }
        })();

        return (
          <div className="app">
            <Sidebar route={route} navigate={navigate} user={user} onLogout={() => { setUser(null); navigate('home'); }}/>
            <main className="main">
              <Topbar title={meta[0]} subtitle={meta[1]} user={user} route={route}/>
              {page}
            </main>
          </div>
        );
      })()}
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
