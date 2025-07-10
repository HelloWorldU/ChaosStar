# build stage
FROM node:20.14.0 AS build

WORKDIR /app

COPY web/frontend/package*.json ./
COPY web/frontend/tsconfig.json ./
COPY web/shared/ ../shared/

RUN npm install

COPY web/frontend/ ./

# RUN ln -sf ./shared ../shared

RUN npm run build


# release stage
FROM nginx:stable-alpine AS release

RUN rm /etc/nginx/conf.d/default.conf

COPY docker/nginx.conf /etc/nginx/conf.d/chaosstar.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
