/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Particle System - Beautiful Animated Background
 * ═══════════════════════════════════════════════════════════════════════════
 */

export class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.connections = [];
        this.mouse = { x: null, y: null, radius: 150 };

        this.config = {
            particleCount: 50,
            particleMinSize: 1,
            particleMaxSize: 2,
            particleSpeed: 0.2,
            connectionDistance: 120,
            connectionOpacity: 0.08,
            colors: {
                particle: ['#94a3b8', '#cbd5e1', '#64748b', '#a1a1aa'],
                connection: '#94a3b8'
            }
        };

        this.init();
        this.bindEvents();
    }

    init() {
        this.resize();
        this.createParticles();
        this.animate();
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);

        this.width = rect.width;
        this.height = rect.height;
    }

    bindEvents() {
        window.addEventListener('resize', () => {
            this.resize();
            this.createParticles();
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouse.x = e.clientX - rect.left;
            this.mouse.y = e.clientY - rect.top;
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.mouse.x = null;
            this.mouse.y = null;
        });
    }

    createParticles() {
        this.particles = [];

        for (let i = 0; i < this.config.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                vx: (Math.random() - 0.5) * this.config.particleSpeed,
                vy: (Math.random() - 0.5) * this.config.particleSpeed,
                size: this.config.particleMinSize +
                    Math.random() * (this.config.particleMaxSize - this.config.particleMinSize),
                color: this.config.colors.particle[
                    Math.floor(Math.random() * this.config.colors.particle.length)
                ],
                pulse: Math.random() * Math.PI * 2,
                pulseSpeed: 0.02 + Math.random() * 0.02
            });
        }
    }

    updateParticles() {
        for (const p of this.particles) {
            // Update position
            p.x += p.vx;
            p.y += p.vy;

            // Update pulse animation
            p.pulse += p.pulseSpeed;

            // Bounce off edges with smooth deceleration
            if (p.x < 0 || p.x > this.width) {
                p.vx *= -1;
                p.x = Math.max(0, Math.min(this.width, p.x));
            }
            if (p.y < 0 || p.y > this.height) {
                p.vy *= -1;
                p.y = Math.max(0, Math.min(this.height, p.y));
            }

            // Mouse interaction - subtle attraction/repulsion
            if (this.mouse.x !== null && this.mouse.y !== null) {
                const dx = this.mouse.x - p.x;
                const dy = this.mouse.y - p.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.mouse.radius) {
                    const force = (this.mouse.radius - distance) / this.mouse.radius;
                    const angle = Math.atan2(dy, dx);

                    // Gentle repulsion
                    p.vx -= Math.cos(angle) * force * 0.02;
                    p.vy -= Math.sin(angle) * force * 0.02;
                }
            }

            // Speed limit
            const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
            if (speed > this.config.particleSpeed * 2) {
                p.vx = (p.vx / speed) * this.config.particleSpeed * 2;
                p.vy = (p.vy / speed) * this.config.particleSpeed * 2;
            }
        }
    }

    drawParticles() {
        for (const p of this.particles) {
            const pulseScale = 1 + Math.sin(p.pulse) * 0.3;
            const size = p.size * pulseScale;

            // Glow effect
            const gradient = this.ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, size * 3);
            gradient.addColorStop(0, p.color);
            gradient.addColorStop(0.5, p.color + '40');
            gradient.addColorStop(1, 'transparent');

            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, size * 3, 0, Math.PI * 2);
            this.ctx.fillStyle = gradient;
            this.ctx.fill();

            // Core
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;
            this.ctx.fill();
        }
    }

    drawConnections() {
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const p1 = this.particles[i];
                const p2 = this.particles[j];

                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.config.connectionDistance) {
                    const opacity = (1 - distance / this.config.connectionDistance) *
                        this.config.connectionOpacity;

                    // Gradient line
                    const gradient = this.ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
                    gradient.addColorStop(0, p1.color + Math.floor(opacity * 255).toString(16).padStart(2, '0'));
                    gradient.addColorStop(1, p2.color + Math.floor(opacity * 255).toString(16).padStart(2, '0'));

                    this.ctx.beginPath();
                    this.ctx.moveTo(p1.x, p1.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.strokeStyle = gradient;
                    this.ctx.lineWidth = 1;
                    this.ctx.stroke();
                }
            }
        }

        // Connection to mouse
        if (this.mouse.x !== null && this.mouse.y !== null) {
            for (const p of this.particles) {
                const dx = this.mouse.x - p.x;
                const dy = this.mouse.y - p.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.mouse.radius) {
                    const opacity = (1 - distance / this.mouse.radius) * 0.15;

                    this.ctx.beginPath();
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(this.mouse.x, this.mouse.y);
                    this.ctx.strokeStyle = `rgba(100, 116, 139, ${opacity})`;
                    this.ctx.lineWidth = 1;
                    this.ctx.stroke();
                }
            }
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.width, this.height);

        this.updateParticles();
        this.drawConnections();
        this.drawParticles();

        requestAnimationFrame(() => this.animate());
    }

    /**
     * Create burst effect at position (for training events)
     */
    burst(x, y, color = '#64748b') {
        const burstCount = 10;
        for (let i = 0; i < burstCount; i++) {
            const angle = (Math.PI * 2 * i) / burstCount;
            const speed = 2 + Math.random() * 2;

            this.particles.push({
                x,
                y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                size: 2 + Math.random() * 2,
                color,
                pulse: 0,
                pulseSpeed: 0.1,
                lifetime: 60,
                decay: true
            });
        }
    }
}
