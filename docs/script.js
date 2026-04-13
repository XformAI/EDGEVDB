/* ============================================
   EdgeVDB Documentation — Interactive Scripts
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
    // ── Navbar scroll effect ──
    const navbar = document.getElementById('main-nav');
    const backToTop = document.getElementById('back-to-top');
    const scrollHint = document.getElementById('scroll-hint');

    const handleScroll = () => {
        const scrollY = window.scrollY;
        
        if (scrollY > 60) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        if (scrollY > 400) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }

        if (scrollHint && scrollY > 100) {
            scrollHint.style.opacity = Math.max(0, 1 - (scrollY - 100) / 200);
        }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    // ── Back to top ──
    backToTop.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // ── Mobile nav toggle ──
    const navToggle = document.getElementById('nav-toggle');
    const navLinks = document.getElementById('nav-links');

    navToggle.addEventListener('click', () => {
        navLinks.classList.toggle('open');
        navToggle.classList.toggle('active');
    });

    // Close on link click
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('open');
            navToggle.classList.remove('active');
        });
    });

    // ── Active nav highlighting ──
    const sections = document.querySelectorAll('section[id]');
    const navAnchors = navLinks.querySelectorAll('a[href^="#"]');

    const observerNav = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                navAnchors.forEach(a => {
                    a.classList.toggle('active', a.getAttribute('href') === `#${id}`);
                });
            }
        });
    }, { rootMargin: '-40% 0px -60% 0px' });

    sections.forEach(sec => observerNav.observe(sec));

    // ── Install tabs (hero) ──
    const installTabs = document.querySelectorAll('.install-tab');
    const installCodes = document.querySelectorAll('.install-code');

    installTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const lang = tab.dataset.lang;
            installTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            installCodes.forEach(code => {
                code.classList.toggle('hidden', !code.id.includes(lang));
            });
        });
    });

    // ── Copy to clipboard ──
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            let text = '';
            const targetId = btn.dataset.target;
            if (targetId) {
                const el = document.getElementById(targetId);
                if (el) text = el.textContent;
            } else {
                // Install codes — find active one
                const active = document.querySelector('.install-code:not(.hidden)');
                if (active) text = active.textContent;
            }

            if (text) {
                navigator.clipboard.writeText(text.trim()).then(() => {
                    const orig = btn.innerHTML;
                    btn.innerHTML = '<span style="font-size:0.75rem;color:#10b981">✓ Copied</span>';
                    setTimeout(() => btn.innerHTML = orig, 2000);
                });
            }
        });
    });

    // ── Quick Start tabs ──
    const qsTabs = document.querySelectorAll('#quickstart-tabs .tab');
    const qsPanels = document.querySelectorAll('#quickstart-tabs .tab-panel');

    qsTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.target;
            qsTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            qsPanels.forEach(p => {
                p.classList.toggle('active', p.id === target);
            });
        });
    });

    // ── Intersection Observer for reveal animations ──
    const revealElements = document.querySelectorAll(
        '.feature-card, .arch-block, .api-group, .bench-card, .bench-comparison, ' +
        '.platform-card, .build-card, .flow-step, .arch-flow'
    );

    revealElements.forEach(el => el.classList.add('reveal'));

    const revealObserver = new IntersectionObserver(entries => {
        entries.forEach((entry, i) => {
            if (entry.isIntersecting) {
                // Stagger animation
                const siblings = Array.from(entry.target.parentElement.children);
                const index = siblings.indexOf(entry.target);
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, index * 80);
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    revealElements.forEach(el => revealObserver.observe(el));

    // ── Animated stat counters ──
    const statSection = document.querySelector('.hero-stats');
    let hasAnimated = false;

    const animateValue = (el, end, suffix, duration) => {
        const start = 0;
        const startTime = performance.now();
        
        const update = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const ease = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * ease;

            const valueEl = el.querySelector('.stat-value');
            if (valueEl) {
                if (Number.isInteger(end)) {
                    valueEl.innerHTML = Math.round(current) + (suffix ? `<span class="stat-unit">${suffix}</span>` : '');
                } else {
                    valueEl.innerHTML = current.toFixed(end < 10 ? 2 : 1) + (suffix ? `<span class="stat-unit">${suffix}</span>` : '');
                }
            }

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        };

        requestAnimationFrame(update);
    };

    const statObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !hasAnimated) {
                hasAnimated = true;
                animateValue(document.getElementById('stat-latency'), 0.99, 'ms', 1500);
                animateValue(document.getElementById('stat-recall'), 96.8, '%', 1500);
                animateValue(document.getElementById('stat-size'), 428, 'KB', 1500);
                animateValue(document.getElementById('stat-deps'), 0, '', 800);
            }
        });
    }, { threshold: 0.5 });

    if (statSection) statObserver.observe(statSection);

    // ── Benchmark bar animation ──
    const benchBars = document.querySelectorAll('.bench-bar');
    const barObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.width = entry.target.style.getPropertyValue('--bar-width');
                barObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    benchBars.forEach(bar => {
        bar.style.width = '0%';
        barObserver.observe(bar);
    });

    // ── Hero particle canvas ──
    const canvas = document.getElementById('hero-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let width, height;
        const particles = [];
        const particleCount = 60;
        let animFrameId;

        const resize = () => {
            width = canvas.width = canvas.offsetWidth;
            height = canvas.height = canvas.offsetHeight;
        };

        class Particle {
            constructor() {
                this.reset();
            }

            reset() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.3;
                this.vy = (Math.random() - 0.5) * 0.3;
                this.radius = Math.random() * 1.5 + 0.5;
                this.opacity = Math.random() * 0.4 + 0.1;
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(99, 102, 241, ${this.opacity})`;
                ctx.fill();
            }
        }

        const init = () => {
            resize();
            particles.length = 0;
            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
        };

        const drawConnections = () => {
            const maxDist = 150;
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < maxDist) {
                        const opacity = (1 - dist / maxDist) * 0.12;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(99, 102, 241, ${opacity})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
        };

        const animate = () => {
            ctx.clearRect(0, 0, width, height);
            
            particles.forEach(p => {
                p.update();
                p.draw();
            });

            drawConnections();
            animFrameId = requestAnimationFrame(animate);
        };

        init();
        animate();

        window.addEventListener('resize', () => {
            resize();
        });

        // Performance: stop animation when hero is not visible
        const heroSection = document.getElementById('hero');
        const heroObserver = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    if (!animFrameId) animate();
                } else {
                    if (animFrameId) {
                        cancelAnimationFrame(animFrameId);
                        animFrameId = null;
                    }
                }
            });
        });
        heroObserver.observe(heroSection);
    }

    // ── Smooth scroll for anchor links ──
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // ── Keyboard navigation for tabs ──
    document.querySelectorAll('.tabs-bar, .install-tab-bar').forEach(bar => {
        bar.addEventListener('keydown', (e) => {
            const tabs = Array.from(bar.querySelectorAll('button'));
            const current = tabs.findIndex(t => t.classList.contains('active'));

            if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
                e.preventDefault();
                const next = e.key === 'ArrowRight'
                    ? (current + 1) % tabs.length
                    : (current - 1 + tabs.length) % tabs.length;
                tabs[next].click();
                tabs[next].focus();
            }
        });
    });
});
