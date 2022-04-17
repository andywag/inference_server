# set base image (host OS)
FROM nginx

WORKDIR /nginx
# copy the dependencies file to the working directory
COPY nginx_docker.conf .
COPY proxy_params /etc/nginx/proxy_params

# command to run on container start
CMD [ "nginx", "-c","/nginx/nginx_docker.conf","-g", "daemon off;"]