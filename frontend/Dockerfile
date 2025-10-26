FROM nginx:alpine

ARG API_BASE=http://localhost:8000
COPY index.html /usr/share/nginx/html/index.html
# Inject the API_BASE at build time
RUN sed -i "s|http://localhost:8000|${API_BASE}|g" /usr/share/nginx/html/index.html

EXPOSE 80
